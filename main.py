"""
Prompt generation CLI.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.pipeline import run_core_concept_pipeline


def main(config_path: str = "config.yaml", dry_run: bool = False):
    print("=" * 60)
    print("Prompt自动生成框架 V2")
    print("=" * 60)

    print("\n[1/6] 加载配置...")
    config = load_config(config_path)
    print(f"  - 采样模式: {config.sampling_mode}")
    print(f"  - Dry-run: {dry_run}")

    if config.sampling_mode != 'core_concept':
        raise ValueError(
            "旧 template/V1 流程已删除，请将 core_sampling.sampling_mode 设置为 core_concept"
        )

    run_core_concept_pipeline(config, dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt自动生成框架 V2")
    parser.add_argument("--config", "-c", default="config.yaml", help="配置文件路径")
    parser.add_argument("--dry-run", "-d", action="store_true", help="Dry-run模式")
    args = parser.parse_args()
    main(args.config, args.dry_run)
