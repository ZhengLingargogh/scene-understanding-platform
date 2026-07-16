import pytest

from app.db.session import init_db


@pytest.fixture(autouse=True)
def _ensure_db_tables():
    init_db()
