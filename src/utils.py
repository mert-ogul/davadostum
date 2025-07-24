from pathlib import Path
import sqlite3
from contextlib import contextmanager
import itertools


def ensure_dirs():
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)


@contextmanager
def get_connection(db_path: str, timeout: int = 30):
    """Yield a SQLite connection with safer defaults for concurrent access.

    Args:
        db_path: Path to the SQLite database file.
        timeout: Seconds to wait if the database is locked (defaults to 30).
    """
    # Ensure parent directory exists (especially when a custom path like data/*.sqlite is used)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path, timeout=timeout, check_same_thread=False)

    # Enable WAL so multiple readers/writers can coexist more gracefully.
    # This dramatically reduces "database is locked" errors in multi-process scenarios.
    conn.execute("PRAGMA journal_mode=WAL;")
    # Align busy timeout pragma with the python-level timeout for extra safety.
    conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)};")

    try:
        yield conn
    finally:
        conn.close()


# Helper: cycle through a list endlessly
def cycler(seq):
    if not seq:
        while True:
            yield None
    else:
        for item in itertools.cycle(seq):
            yield item


def load_list_file(path: str):
    """Load a text file into a list, ignoring empty/comment lines."""
    p = Path(path)
    if not p.exists():
        return []
    return [line.strip() for line in p.read_text("utf-8").splitlines() if line.strip() and not line.startswith("#")]
