import json
import gzip
from pathlib import Path
from typing import Generator, Dict, Any, Union

def load_candidates(file_path: Union[str, Path]) -> Generator[Dict[str, Any], None, None]:
    """
    Generator that yields candidate dictionaries from a JSONL file.
    Supports both raw .jsonl and gzipped .jsonl.gz files.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")

    # Determine if the file is gzipped
    is_gzip = path.suffix == '.gz' or (len(path.suffixes) > 1 and path.suffixes[-2] == '.jsonl' and path.suffixes[-1] == '.gz')

    if is_gzip:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

def load_candidates_list(file_path: Union[str, Path]) -> list:
    """
    Loads all candidates into a list in memory.
    """
    return list(load_candidates(file_path))
