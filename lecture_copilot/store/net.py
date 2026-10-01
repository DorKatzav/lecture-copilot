"""The one door to the network (DESIGN_HE §engineering, criterion 5). Every outbound call goes through `Net.call`
and leaves a `net` row in `decisions`: host, bytes, tokens, cost, status. Audio never passes here.

A connection failure is `Offline` (the caller marks the item unchecked and tries again later); any other failure
is retried once with backoff and then raised as `NetError`.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable

import httpx

from lecture_copilot.config import GEMINI_PRICE_PER_M
from lecture_copilot.store.db import Store

OFFLINE_ERRORS = (ConnectionError, TimeoutError, httpx.TransportError, OSError)


class NetError(RuntimeError):
    pass


class Offline(NetError):
    pass


def gemini_cost(tokens_in: int, tokens_out: int) -> float:
    return tokens_in * GEMINI_PRICE_PER_M["in"] / 1e6 + tokens_out * GEMINI_PRICE_PER_M["out"] / 1e6


class Net:
    def __init__(self, store: Store, backoff_s: float = 1.0):
        self.store, self.backoff_s = store, backoff_s

    async def call(self, host: str, fn: Callable[[], Awaitable[tuple[object, dict]]], *, lecture_id: str | None,
                   ref: str) -> object:
        """`fn` returns (result, usage) with usage = {bytes_out, bytes_in, tokens_in, tokens_out, cost_usd}."""
        t0 = time.perf_counter()
        last: Exception | None = None
        for attempt in (1, 2):
            try:
                result, usage = await fn()
            except OFFLINE_ERRORS as e:
                self._log(host, lecture_id, ref, t0, {"status": "offline", "attempts": attempt,
                                                       "error": f"{type(e).__name__}: {e}"})
                raise Offline(f"{host}: {e}") from e
            except Exception as e:
                last = e
                if attempt == 1:
                    await asyncio.sleep(self.backoff_s)
                continue
            self._log(host, lecture_id, ref, t0, {"status": "ok", "attempts": attempt, **usage},
                      tokens_in=usage.get("tokens_in"), tokens_out=usage.get("tokens_out"),
                      cost_usd=usage.get("cost_usd"))
            return result
        self._log(host, lecture_id, ref, t0, {"status": "failed", "attempts": 2,
                                               "error": f"{type(last).__name__}: {last}"})
        raise NetError(f"{host}: {last}") from last

    def _log(self, host: str, lecture_id: str | None, ref: str, t0: float, out: dict, **cols) -> None:
        self.store.log("net", lecture_id=lecture_id, input_ref=ref, ms=(time.perf_counter() - t0) * 1000,
                       output={"host": host, **out}, **cols)

    def cost(self, lecture_id: str) -> float:
        row = self.store.con.execute("select coalesce(sum(cost_usd), 0) from decisions where node = 'net' "
                                     "and lecture_id = ?", (lecture_id,)).fetchone()
        return float(row[0])

    def calls(self, lecture_id: str) -> int:
        return self.store.con.execute("select count(*) from decisions where node = 'net' and lecture_id = ?",
                                      (lecture_id,)).fetchone()[0]
