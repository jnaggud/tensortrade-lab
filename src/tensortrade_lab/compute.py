"""One global CPU budget: parallel trials OR parallel environments, never both uncontrolled."""

import os
import platform
from dataclasses import asdict, dataclass

import psutil


def hardware():
    import torch

    return {
        "platform": platform.platform(),
        "cpu_cores": os.cpu_count() or 1,
        "physical_cores": psutil.cpu_count(logical=False),
        "memory_gib": round(psutil.virtual_memory().total / 2**30, 1),
        "available_memory_gib": round(psutil.virtual_memory().available / 2**30, 1),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_devices": torch.cuda.device_count(),
        "torch_version": torch.__version__,
    }


def resolve_device(requested):
    import torch

    if requested in ("auto", "hybrid"):
        # Small MLP PPO is often CPU-bound. Searches separately schedule GPU trial slots.
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError(
            "Metal/MPS is unavailable in this process; use CPU or run outside a restricted sandbox"
        )
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but no CUDA device is available")
    return requested


def configure_threads(count=1):
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
    ):
        os.environ[name] = str(count)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(count)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # May already be initialized in a reused worker.
    threadpool_limits(limits=count)


@dataclass
class ResourcePlan:
    cpu_budget: int
    worker_count: int
    threads_per_worker: int
    devices: list[str]
    memory_limit_workers: int

    def to_dict(self):
        return asdict(self)


def plan_resources(config, jobs: int, info=None):
    info = info or hardware()
    cores = info["cpu_cores"]
    threads = config.threads_per_worker
    if threads > cores:
        raise ValueError("threads_per_worker exceeds the CPU budget")
    requested = config.workers or max(1, cores // threads)
    memory_limit = max(1, int(info["available_memory_gib"] // 2))
    workers = min(requested, max(1, cores // threads), memory_limit, max(1, jobs))
    devices = ["cpu"] * workers
    if config.device in ("mps", "cuda"):
        available = info["mps_available"] if config.device == "mps" else info["cuda_devices"] > 0
        if not available:
            raise ValueError(f"Requested {config.device} is unavailable")
        workers = min(workers, config.gpu_jobs * max(1, info["cuda_devices"]))
        devices = [
            config.device if config.device == "mps" else f"cuda:{i % info['cuda_devices']}"
            for i in range(workers)
        ]
    elif config.device in ("hybrid", "auto"):
        gpu = "mps" if info["mps_available"] else ("cuda:0" if info["cuda_devices"] else None)
        if gpu and workers > 1:
            for i in range(min(config.gpu_jobs * max(1, info["cuda_devices"]), workers - 1)):
                devices[i] = gpu if gpu == "mps" else f"cuda:{i % info['cuda_devices']}"
    return ResourcePlan(cores, workers, threads, devices, memory_limit)


def benchmark(output):
    """Measured forward/backward throughput; not a claim about end-to-end PPO speed."""
    import time
    from pathlib import Path

    import torch

    from .artifacts import write_json

    detected = hardware()
    devices = (
        ["cpu"]
        + (["mps"] if detected["mps_available"] else [])
        + [f"cuda:{i}" for i in range(detected["cuda_devices"])]
    )
    results = []
    for width, batch in ((64, 128), (1024, 4096)):
        for device in devices:
            for threads in [1, detected["cpu_cores"]] if device == "cpu" else [1]:
                configure_threads(threads)
                model = torch.nn.Sequential(
                    torch.nn.Linear(width, width), torch.nn.Tanh(), torch.nn.Linear(width, 10)
                ).to(device)
                x = torch.randn(batch, width, device=device)

                def sync(device=device):
                    if device == "mps":
                        torch.mps.synchronize()
                    elif device.startswith("cuda"):
                        torch.cuda.synchronize(device)

                for _ in range(3):
                    model.zero_grad(set_to_none=True)
                    model(x).square().mean().backward()
                sync()
                start = time.perf_counter()
                for _ in range(20):
                    model.zero_grad(set_to_none=True)
                    model(x).square().mean().backward()
                sync()
                results.append(
                    {
                        "width": width,
                        "batch": batch,
                        "device": device,
                        "cpu_threads": threads,
                        "steps_per_second": 20 / (time.perf_counter() - start),
                    }
                )
    report = {
        "hardware": detected,
        "results": results,
        "note": "Kernel benchmark. CPU simulations remain on CPU; Metal/CUDA accelerates tensor operations. Measure full trial duration before choosing GPU for small MLPs.",
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, report)
    configure_threads(1)
    return report


def parallel_jobs(function, jobs, config):
    """Run independent jobs in resource slots; each slot owns at most one GPU job."""
    import multiprocessing as mp
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    plan = plan_resources(config, len(jobs))
    configure_threads(plan.threads_per_worker)
    pools = [ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) for _ in plan.devices]
    pending, results, cursor = {}, {}, 0
    available = list(range(len(pools)))
    try:
        while pending or cursor < len(jobs):
            while available and cursor < len(jobs):
                slot = available.pop(0)
                pending[
                    pools[slot].submit(function, jobs[cursor], plan.devices[slot], plan.threads_per_worker)
                ] = (slot, cursor)
                cursor += 1
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                slot, index = pending.pop(future)
                results[index] = future.result()
                available.append(slot)
    finally:
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=True)
    return [results[i] for i in range(len(jobs))]
