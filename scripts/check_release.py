"""Check the public file set without reading ignored local research archives."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{60,})"),
    "AWS access key": re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    "service token": re.compile(
        r"(?:sk-[A-Za-z0-9_-]{24,}|xox[baprs]-[A-Za-z0-9-]{20,}|AIza[A-Za-z0-9_-]{30,})"
    ),
    "literal credential": re.compile(
        r"(?:api[_-]?key|secret|access[_-]?token|password)\s*[:=]\s*['\"][A-Za-z0-9_=/+-]{16,}['\"]",
        re.IGNORECASE,
    ),
    "personal filesystem path": re.compile(r"/(?:Users|Volumes)/[A-Za-z0-9_.-]+/"),
    "private layout": re.compile(r"https://www\.tradingview\.com/chart/[A-Za-z0-9]+/"),
}
PRIVATE_SUFFIXES = {".parquet", ".dbn", ".zst", ".zip", ".db", ".sqlite", ".pem", ".key", ".log"}


def public_files():
    result = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    )
    return sorted({Path(p.decode()) for p in result.split(b"\0") if p})


def main():
    paths = public_files()
    problems = []
    total = 0
    for relative in paths:
        path = ROOT / relative
        if path.is_symlink():
            problems.append(f"{relative}: unexpected symlink")
            continue
        raw = path.read_bytes()
        total += len(raw)
        if len(raw) > 5 * 1024 * 1024:
            problems.append(f"{relative}: file exceeds the 5 MiB source-release limit")
        if path.suffix in PRIVATE_SUFFIXES or path.name.startswith(".env"):
            problems.append(f"{relative}: private/generated file type")
        if path.suffix == ".py":
            try:
                compile(raw, str(relative), "exec")
            except SyntaxError as exc:
                problems.append(f"{relative}: invalid Python at line {exc.lineno}")
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(content):
                problems.append(f"{relative}: {label}")
        if path.suffix == ".md":
            for target in re.findall(r"\]\(([^\s)]+)(?:\s+[^)]*)?\)", content):
                if target.startswith(("https://", "http://", "mailto:", "#")):
                    continue
                local = target.split("#", 1)[0]
                destination = (path.parent / local).resolve()
                if not destination.is_relative_to(ROOT) or not destination.exists():
                    problems.append(f"{relative}: broken local link {target}")
                elif destination.is_file() and destination.relative_to(ROOT) not in paths:
                    problems.append(f"{relative}: link to excluded file {target}")
    if problems:
        raise SystemExit("Release checks failed:\n" + "\n".join(problems))
    print(f"Release checks passed: {len(paths)} files, {total / 1024 / 1024:.2f} MiB")
    print("Python sources compile; local links resolve; no targeted private-data patterns found.")


if __name__ == "__main__":
    main()
