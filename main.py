"""
统一入口脚本（示例）

你可以在这里根据命令行参数路由到不同实验脚本，
例如：
    python main.py --exp traffic
    python main.py --exp harmonic_vp
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="LCF Clean Main Entry")
    parser.add_argument(
        "--exp",
        type=str,
        default="traffic",
        choices=["traffic", "harmonic_vp", "improved_encoder", "catsg_benchmark"],
        help="选择要运行的实验",
    )
    args, unknown = parser.parse_known_args()

    module_map = {
        "traffic": "lcf.scripts.experiments.test_traffic",
        "harmonic_vp": "lcf.scripts.experiments.test_harmonic_vp",
        "improved_encoder": "lcf.scripts.experiments.test_improved_encoder",
        "catsg_benchmark": "lcf.scripts.experiments.catsg_benchmark",
    }

    target_module = module_map[args.exp]

    # 将剩余参数原样传给目标脚本
    cmd = [sys.executable, "-m", target_module, *unknown]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

