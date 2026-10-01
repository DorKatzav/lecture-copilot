"""Hybrid search over the course memory (DESIGN_HE §memory): FTS5 bm25 + vector cosine, fused by RRF.

The text index lives in two FTS5 tables kept in step with items/claims by the Store. Vectors are the float32 blobs
on the rows; `SqliteVec` keeps a copy in sqlite-vec vec0 tables for KNN, `NumpyVec` scans the course's blobs in
memory (the fallback the spec asked for). Both answer the same question and pass the same tests.
"""

import re
import sqlite3
from dataclasses import dataclass

import numpy as np

from lecture_copilot.store.embed import pack, unpack

RRF_K = 60
_TOKEN = re.compile(r"[\w'׳״]+", re.UNICODE)


@dataclass(frozen=True)
class MemoryHit:
    kind: str                # concept | claim | question | action | highlight | note | decision
    id: str
    lecture_id: str
    text: str
    score: float
    explanation: str | None = None
    canonical_key: str | None = None
    week: int | None = None
    title: str | None = None
    importance: int | None = None


def fts_query(text: str) -> str:
    """Every token quoted and OR-ed: user text never reaches FTS5's query syntax."""
    tokens = [t for t in _TOKEN.findall(text) if t.strip("'")]
    return " OR ".join(f'"{t}"' for t in tokens)


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Ties keep the order of first appearance: the lists are passed items before claims, text before vectors."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, 1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    order = {key: i for i, key in enumerate(scores)}
    return sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))


# ---------- vector backends ----------

class NumpyVec:
    name = "numpy"

    def __init__(self, con: sqlite3.Connection):
        self.con = con

    def upsert(self, table: str, row_id: str, vec: list[float]) -> None:
        pass  # the blob on the row is the index

    def delete(self, table: str, ids: list[str]) -> None:
        pass

    def knn(self, table: str, query: list[float], lecture_ids: list[str], k: int) -> list[tuple[str, float]]:
        if not lecture_ids:
            return []
        marks = ",".join("?" * len(lecture_ids))
        rows = self.con.execute(f"select id, embedding from {table} where embedding is not null "
                                f"and lecture_id in ({marks})", lecture_ids).fetchall()
        if not rows:
            return []
        m = np.array([unpack(r[1]) for r in rows], dtype=np.float32)
        q = np.array(query, dtype=np.float32)
        norms = np.linalg.norm(m, axis=1) * (np.linalg.norm(q) or 1.0)
        sims = (m @ q) / np.where(norms == 0, 1.0, norms)
        order = np.argsort(-sims)[:k]
        return [(rows[i][0], float(sims[i])) for i in order]


class SqliteVec:
    name = "sqlite-vec"

    def __init__(self, con: sqlite3.Connection, dims: int):
        import sqlite_vec
        con.enable_load_extension(True)
        sqlite_vec.load(con)
        con.enable_load_extension(False)
        self.con, self.dims = con, dims
        for table in ("items", "claims"):
            con.execute(f"create virtual table if not exists vec_{table} using vec0(id text primary key, "
                        f"embedding float[{dims}])")

    def upsert(self, table: str, row_id: str, vec: list[float]) -> None:
        self.con.execute(f"delete from vec_{table} where id = ?", (row_id,))
        self.con.execute(f"insert into vec_{table} (id, embedding) values (?, ?)", (row_id, pack(vec)))

    def delete(self, table: str, ids: list[str]) -> None:
        self.con.executemany(f"delete from vec_{table} where id = ?", [(i,) for i in ids])

    def knn(self, table: str, query: list[float], lecture_ids: list[str], k: int) -> list[tuple[str, float]]:
        if not lecture_ids:
            return []
        marks = ",".join("?" * len(lecture_ids))
        rows = self.con.execute(
            f"select v.id, v.distance from vec_{table} v join {table} t on t.id = v.id "
            f"where v.embedding match ? and k = ? and t.lecture_id in ({marks}) order by v.distance",
            (pack(query), max(k * 20, 50), *lecture_ids)).fetchall()
        return [(r[0], 1.0 - r[1] ** 2 / 2) for r in rows[:k]]   # vec0 distance is L2 on unit vectors


def make_backend(con: sqlite3.Connection, dims: int, preferred: str | None) -> NumpyVec | SqliteVec:
    if preferred != "numpy":
        try:
            return SqliteVec(con, dims)
        except Exception:
            if preferred == "sqlite-vec":
                raise
    return NumpyVec(con)
