from pathlib import Path
import tomli
from functools import lru_cache

CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.toml"

@lru_cache
def load_config() -> dict:
    with open(CONFIG_FILE, "rb") as f:
        return tomli.load(f)
