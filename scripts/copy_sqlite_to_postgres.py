"""Copy every row of an existing SQLite database into a PostgreSQL database (one-off move to pgvector).

    .venv/bin/python scripts/copy_sqlite_to_postgres.py sqlite:///data/evaluation.db "postgresql+psycopg://agent@/agent?host=...&port=..."

- Runs the Alembic migrations on the target first (so both have the same schema).
- Copies tables in foreign-key order; rows whose primary key already exists in the target are skipped, so the script
  can be re-run safely (e.g. once before switching DATABASE_URL and once after, to pick up stragglers).
- Resets PostgreSQL id sequences afterwards. The source is only read, never modified.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pgvector.sqlalchemy  # noqa: F401  (registers the "vector" type for reflection)
from sqlalchemy import MetaData, create_engine, func, select, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.db import run_migrations  # noqa: E402


def main(src_url: str, dst_url: str) -> None:
    run_migrations(dst_url)
    src, dst = create_engine(src_url), create_engine(dst_url)
    src_meta, dst_meta = MetaData(), MetaData()
    src_meta.reflect(src)
    dst_meta.reflect(dst)
    order = [t.name for t in dst_meta.sorted_tables if t.name != "alembic_version" and t.name in src_meta.tables]
    with src.connect() as s, dst.begin() as d:
        for name in order:
            st, dt = src_meta.tables[name], dst_meta.tables[name]
            pk = [c.name for c in dt.primary_key.columns]
            existing = {tuple(r) for r in d.execute(select(*[dt.c[c] for c in pk]))}
            cols = [c.name for c in dt.columns if c.name in st.c]
            rows = [dict(r._mapping) for r in s.execute(select(*[st.c[c] for c in cols]))]
            new = [r for r in rows if tuple(r[c] for c in pk) not in existing]
            if new:
                d.execute(dt.insert(), new)
            print(f"{name:<22} source={len(rows):>6}  copied={len(new):>6}  skipped={len(rows) - len(new):>6}")
            # keep autoincrement sequences ahead of the copied ids
            for col in dt.primary_key.columns:
                if col.autoincrement is True or (col.type.python_type is int and len(pk) == 1):
                    seq = d.execute(text("SELECT pg_get_serial_sequence(:t, :c)"), {"t": name, "c": col.name}).scalar()
                    if seq:
                        top = d.execute(select(func.max(dt.c[col.name]))).scalar()
                        if top:
                            d.execute(text("SELECT setval(:s, :v)"), {"s": seq, "v": top})


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
