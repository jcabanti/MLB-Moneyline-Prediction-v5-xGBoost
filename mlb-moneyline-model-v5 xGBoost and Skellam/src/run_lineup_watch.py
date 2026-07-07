from __future__ import annotations

from datetime import date
import argparse

from .run_daily_report import run_daily_report


def run_lineup_watch(game_date: str | None = None):
    # Practical first version: refresh odds and model report.
    # Add lineup-source parsing later when lineup-strength features are implemented.
    return run_daily_report(game_date or date.today().isoformat())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None)
    args = parser.parse_args()
    run_lineup_watch(args.date)
