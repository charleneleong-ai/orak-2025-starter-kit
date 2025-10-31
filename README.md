# ORAK 2025 Starter Kit

Starter template for building and testing Orak 2025 competition agents.

## Quick Start

- Set the required API token: `export AICROWD_API_TOKEN=<your token>` (retrieve your token from [your AIcrowd profile page](https://www.aicrowd.com/participants/me)).
- Run the starter kit from the repository root:

```bash
PYTHONPATH=`pwd` uv run python run.py
```

## Build Your Own Agent

Use the provided random agent as a minimal reference:

```1:6:agents/random.py
class RandomAgent:
    def __init__(self):
        pass
    
    def act(self, obs):
        return "left"
```

- Create your own agent class (new file or modify `agents/random.py`) with an `act` method. The runner passes in the latest observation object from the game environment, and your method must return a valid action string.
- Point `UserAgent` to your implementation by editing `agents/config.py`, for example: `from agents.my_agent import MyAgent` followed by `UserAgent = MyAgent`.
- Run `PYTHONPATH=`pwd` uv run python run.py` again to evaluate your agent.

## Example Output

Example (abbreviated) console output when running the starter kit:

```text
┌───────────────────────────────────────────────┐
│                 Orak Runner                   │
└───────────────────────────────────────────────┘

Session created: 2c9f1a0e-1234-5678-9abc-def012345678
Submission ID: 987654

Status: Game server has started

────────────────────────────────────────────────
Starting game twenty_fourty_eight
────────────────────────────────────────────────

Results: twenty_fourty_eight
Metric   Value
------   -----
Score    64

┌───────────────────────────────────────────────┐
│                    Final                      │
│                 Score: 64                     │
└───────────────────────────────────────────────┘
```

Repeat the edit–run cycle until your agent behaves as desired.
