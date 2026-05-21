from . import v1

# Ordered list — last entry is always the default (latest)
_VERSIONS: list = [v1]

_REGISTRY: dict[str, any] = {
    "v1": v1,
}

LATEST: str = "v1"


def get_prompt(
    matrix: list[list[float]],
    dimension: int,
    intermediate_steps: list[dict],
    version: str = LATEST,
) -> str:
    """Return a formatted innovation prompt string for the given matrix, dimension,
    and pre-computed intermediate steps.

    Args:
        matrix:             The input matrix as a 2D list of floats.
        dimension:          Matrix dimension (n for an n×n matrix).
        intermediate_steps: Pre-computed steps from dataset.json.
        version:            Prompt version key (default: latest).
    """
    module = _REGISTRY.get(version)
    if module is None:
        available = list(_REGISTRY.keys())
        raise ValueError(f"Unknown prompt version '{version}'. Available: {available}")
    return module.prompt(matrix, dimension, intermediate_steps)


def list_versions() -> list[str]:
    """Return all registered version keys in registration order."""
    return [m.__name__.split(".")[-1] for m in _VERSIONS]
