from __future__ import annotations

import re
import pytest


@pytest.mark.smoke
def test_import_and_version() -> None:
    """Verify that `my_swampe` imports and exposes a version string."""
    # JAX is a required runtime dependency for the numerical core.
    import jax  # noqa: F401  # pylint: disable=unused-import
    import my_swampe  # noqa: F401  # pylint: disable=unused-import

    assert hasattr(my_swampe, "__version__")
    assert re.match(r"^\d+\.\d+\.\d+$", my_swampe.__version__), "Version should look like x.y.z"
