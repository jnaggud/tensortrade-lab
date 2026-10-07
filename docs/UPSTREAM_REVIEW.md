# TensorTrade source and documentation review

Reviewed the `tensortrade-org/tensortrade` default branch at commit
`d58afba23deb1fded39793203b7997c3990bb032` before implementation.

Primary sources:

- [Repository README](https://github.com/tensortrade-org/tensortrade/blob/d58afba23deb1fded39793203b7997c3990bb032/README.md)
- [Architecture](https://github.com/tensortrade-org/tensortrade/blob/d58afba23deb1fded39793203b7997c3990bb032/docs/tutorials/01-foundations/02-architecture.md)
- [Action schemes](https://github.com/tensortrade-org/tensortrade/blob/d58afba23deb1fded39793203b7997c3990bb032/docs/tutorials/03-components/01-action-schemes.md)
- [Rewards](https://github.com/tensortrade-org/tensortrade/blob/d58afba23deb1fded39793203b7997c3990bb032/docs/tutorials/03-components/02-reward-schemes.md)
- [Observers and feeds](https://github.com/tensortrade-org/tensortrade/blob/d58afba23deb1fded39793203b7997c3990bb032/docs/tutorials/03-components/03-observers-feeds.md)
- [Training](https://github.com/tensortrade-org/tensortrade/blob/d58afba23deb1fded39793203b7997c3990bb032/docs/tutorials/04-training/01-first-training.md)
- [Commission analysis](https://github.com/tensortrade-org/tensortrade/blob/d58afba23deb1fded39793203b7997c3990bb032/docs/tutorials/05-advanced/02-commission.md)
- [Walk-forward validation](https://github.com/tensortrade-org/tensortrade/blob/d58afba23deb1fded39793203b7997c3990bb032/docs/tutorials/05-advanced/03-walk-forward.md)

Inspected implementation of generic TradingEnv, default actions, component contracts,
DataFeed, Exchange, simulated order service, Broker, Order construction, Wallet settlement,
and Portfolio reset. This system uses those actual APIs, without copying or replacing
TensorTrade's source tree.

## Findings that affected implementation

1. The current source enforces Python >=3.12. Older install snippets and package releases
   may describe other versions. This project targets Python 3.12 and pins the reviewed commit.
2. The README's `tensortrade.org` documentation URL redirected to unrelated casino content
   when accessed during this task. Repository-hosted documentation was used instead.
3. Some tutorial action descriptions disagree with the code. BSH exposes `Discrete(2)`;
   the simple example attempts action 2 as a hold. We use an explicit six-action component
   with a tested action mapping instead of relying on those examples.
4. Default PBR is a direction-oriented learning signal and does not directly represent
   commission-net portfolio change. This project rewards realized account equity changes,
   including costs and held-position mark-to-market.
5. Some example feature pipelines use backfilling. That can leak future values into
   early rows. This project drops warmup rows and fits its scaler on training only.
6. `TradingEnv.reset` does not initialize Gymnasium's seeded RNG. ResearchEnv initializes
   it before invoking TensorTrade reset and passes the Gym/SB3 environment check.
7. The simulated execution service uses Decimal quotes. Order prices use the exact same
   Decimal quote to avoid a float/Decimal division error in the upstream buy path.
8. Portfolio accounting and feed progression need explicit timing. This system sets an
   execution-open quote for OMS settlement and values balances at close for the reward,
   preventing an action from filling at the same close that produced its indicators.
9. Documentation's performance examples are upstream experimental claims, not evidence
   for this model. Their results are not reproduced or presented as our performance.

Stable-Baselines3 PPO was selected as a single-process Gymnasium-compatible trainer.
Ray is optional in TensorTrade's examples, not required by the component interfaces.
Avoiding a distributed runtime makes local execution and reproducibility simpler.
