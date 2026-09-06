import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def manifest_df():
    from physio.data.manifest import build_manifest
    return build_manifest()


@pytest.fixture()
def tmp_db(tmp_path):
    from physio.db.models import make_session_factory
    return make_session_factory(f"sqlite:///{tmp_path / 'test.db'}")
