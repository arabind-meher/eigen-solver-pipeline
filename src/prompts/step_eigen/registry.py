from . import v1, v2

# Ordered list — last entry is always the default (latest)
_VERSIONS: list = [v1, v2]

_REGISTRY: dict[str, any] = {
    "v1": v1,
    "v2": v2,
}

LATEST: str = "v2"


def get_prompt(
    matrix: list[list[float]],
    dimension: int,
    intermediate_steps: list[dict],
    eigenvectors: list[list[float]],
    version: str = LATEST,
) -> str:
    """Return a formatted prompt string for the given matrix, dimension, intermediate steps, and verified eigenvectors."""
    module = _REGISTRY.get(version)
    if module is None:
        available = list(_REGISTRY.keys())
        raise ValueError(f"Unknown prompt version '{version}'. Available: {available}")
    return module.prompt(matrix, dimension, intermediate_steps, eigenvectors)


def list_versions() -> list[str]:
    """Return all registered version keys in registration order."""
    return [m.__name__.split(".")[-1] for m in _VERSIONS]
