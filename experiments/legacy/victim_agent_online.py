# victim_agent_online.py
# 兼容旧入口：实际无防御攻击复现实验入口已迁移到 experiments/run_attack_reproduction.py。

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_attack_reproduction import main


if __name__ == "__main__":
    main()
