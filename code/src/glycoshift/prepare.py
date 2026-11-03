from __future__ import annotations

import argparse

from .config import load_config, resolve_path, save_config
from .io import write_json
from .pipeline import prepare_dataset
from .streaming import prepare_window_shards


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare normalized CGM windows")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="write participant window shards without materializing all windows",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = resolve_path(config, config["output"]["run_dir"])
    if run_dir is None:
        raise ValueError("output.run_dir is required")
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.streaming:
        input_path = resolve_path(config, config["data"].get("input_path"))
        if input_path is None or not input_path.exists():
            raise FileNotFoundError(
                "normalized input is required for streaming prepare"
            )
        audit = prepare_window_shards(
            input_path,
            run_dir / "window_shards",
            config=config,
            include_holdouts=False,
        )
        write_json(audit, run_dir / "cohort_audit.json")
        save_config(config, run_dir / "config_resolved.yaml")
        print(
            f"prepared {audit['windows_written']} sharded windows in {run_dir}"
        )
        return
    frame, windows, audit = prepare_dataset(
        config,
        include_holdouts=False,
    )
    frame.to_parquet(run_dir / "normalized_input.parquet", index=False)
    windows.to_parquet(run_dir / "prepared_windows.parquet", index=False)
    write_json(audit, run_dir / "cohort_audit.json")
    save_config(config, run_dir / "config_resolved.yaml")
    print(f"prepared {len(windows)} windows in {run_dir}")


if __name__ == "__main__":
    main()
