"""Answer questions about PX4 flight logs with cited, timestamped findings."""


def main() -> None:
    from flightcause.cli import main as cli

    raise SystemExit(cli())
