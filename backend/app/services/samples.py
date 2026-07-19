from functools import lru_cache
from pathlib import Path

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


@lru_cache(maxsize=None)
def load_sample(filename: str) -> str:
    """Read a gold-standard few-shot sample file from app/samples/. Cached
    since these are static, version-controlled files read repeatedly across
    requests."""
    return (SAMPLES_DIR / filename).read_text(encoding="utf-8")


def few_shot_block(*filenames: str) -> str:
    """Build one '# Example: <name>\\n<content>' block per file, concatenated,
    ready to splice directly into a system prompt."""
    return "\n\n".join(f"# Example: {name}\n{load_sample(name)}" for name in filenames)
