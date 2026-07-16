import enum


class SceneStatus(str, enum.Enum):
    """Whether the scene feature index has been loaded."""

    unloaded = "unloaded"
    loading = "loading"
    loaded = "loaded"
    error = "error"
