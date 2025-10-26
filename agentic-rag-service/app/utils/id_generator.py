"""ID generation utilities compatible with LibreChat."""

from nanoid import generate


def generate_id(size: int = 21) -> str:
    """
    Generate a unique ID compatible with LibreChat format.

    Args:
        size: Length of ID (default 21 characters)

    Returns:
        Unique ID string
    """
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    return generate(alphabet, size)


def generate_file_id() -> str:
    """Generate a file ID."""
    return f"file_{generate_id(21)}"


def generate_session_id() -> str:
    """Generate a session ID."""
    return f"sess_{generate_id(21)}"


def generate_chunk_id() -> str:
    """Generate a chunk ID."""
    return f"chunk_{generate_id(21)}"
