import sys
from importlib import import_module
from pathlib import Path


def _bootstrap() -> int:
    root = Path(__file__).resolve().parent
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    cli = import_module("uniter.cli")
    return cli.main()


if __name__ == "__main__":
    raise SystemExit(_bootstrap())
