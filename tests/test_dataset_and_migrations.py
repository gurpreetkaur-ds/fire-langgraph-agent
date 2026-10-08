import sqlalchemy as sa

from evaluation.dataset import load_dataset
from evaluation.db import run_migrations


def test_dataset_is_valid_and_substantial():
    ds = load_dataset()
    assert len(ds.cases) >= 20
    assert len({c.id for c in ds.cases}) == len(ds.cases)
    cats = {c.category for c in ds.cases}
    assert {"safety", "multi-part", "security"} <= cats
    assert any(c.expect_refusal for c in ds.cases)
    assert all(len(c.question) > 15 for c in ds.cases)


def test_migrations_create_all_tables_and_are_repeatable(tmp_path):
    url = f"sqlite:///{tmp_path / 'm.db'}"
    run_migrations(url)
    run_migrations(url)  # idempotent
    tables = set(sa.inspect(sa.create_engine(url)).get_table_names())
    assert {"agent_traces", "agent_steps", "tool_calls", "evaluation_runs", "evaluation_metrics", "alembic_version"} <= tables
