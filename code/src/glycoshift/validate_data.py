from __future__ import annotations

import argparse
import json

from .config import load_config
from .pipeline import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CGM schema and splits")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    _, _, audit = prepare_dataset(config, include_holdouts=False)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
