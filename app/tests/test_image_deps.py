#!/usr/bin/env python3
"""The dependency set the image actually needs at runtime.

These are not unit tests; they are the checks that would have caught a broken
install. The one that prompted the file: `pyarrow>=7.0.0` was unpinned while
numpy was pinned to 1.24.3, and pyarrow 18+ declares no numpy requirement at
all -- so pip installed 26.0.0, the build passed, the pod reported healthy, and
every parquet read in the image raised

    ImportError: pyarrow requires NumPy 2.0 or newer, found 1.24.3

Nothing in this suite noticed, because the tests that touch parquet stub the
read. The whole Coverage Report shipped and served "no charting" on every route
for that reason alone. A dependency that only fails when it is used wants a
test that uses it.
"""
import io

import pandas as pd
import pytest


def test_pyarrow_imports():
    """pandas reads parquet through pyarrow; a version mismatch with the pinned
    numpy only shows up here, at import, not at install."""
    pytest.importorskip("numpy")
    import pyarrow

    assert pyarrow.__version__


def test_pandas_has_a_parquet_engine():
    """`pd.read_parquet` with no usable engine is an ImportError, which callers
    tend to catch as "the data is not published yet"."""
    frame = pd.DataFrame({"season": [2026], "team": ["DAL"], "rate": [0.61]})
    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    buffer.seek(0)
    back = pd.read_parquet(buffer)
    assert back.to_dict(orient="records") == frame.to_dict(orient="records")


def test_polars_converts_to_pandas():
    """nflreadpy returns polars, and every caller in this app immediately calls
    `.to_pandas()` -- which goes through pyarrow too."""
    pl = pytest.importorskip("polars")
    got = pl.DataFrame({"team": ["DAL", "PHI"], "n": [1, 2]}).to_pandas()
    assert list(got.columns) == ["team", "n"]
    assert len(got) == 2
