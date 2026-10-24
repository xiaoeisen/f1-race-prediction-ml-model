# Share rate-limited, cached HTTP requests across external data sources.
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path

CACHE = Path("cache") / "http"
UA = "f1-predictor/1.0 (personal research project)"
EMPTY_TTL = 3600  # empty answers (data not published yet?) are re-checked after an hour

_urlopen = urllib.request.urlopen  # replaced in tests


class SourceError(Exception):
    pass


class Client:
    def __init__(self, name, base, headers=None, per_second=None, per_minute=None, per_hour=None, timeout=20):
        self.name, self.base, self.headers, self.timeout = name, base.rstrip("/"), dict(headers or {}), timeout
        self.limits = [(w, n) for w, n in ((1.0, per_second), (60.0, per_minute), (3600.0, per_hour)) if n]
        self.stamps = deque()

    def _wait(self):
        while True:
            now = time.monotonic()
            while self.stamps and now - self.stamps[0] > 3600:
                self.stamps.popleft()
            delay = 0.0
            for window, n in self.limits:
                recent = [t for t in self.stamps if now - t < window]
                if len(recent) >= n:
                    delay = max(delay, window - (now - recent[-n]))
            if delay <= 0:
                break
            time.sleep(min(delay, 60) + 0.01)
        self.stamps.append(time.monotonic())

    def get(self, path, params=None, ttl=None, allow_404=False, headers=None):
        """GET -> parsed JSON. ttl=None caches forever (historical data)."""
        url = self.base + path + ("?" + urllib.parse.urlencode(params, doseq=True) if params else "")
        f = CACHE / self.name / (hashlib.sha1(url.encode()).hexdigest() + ".json")
        if f.exists():
            try:
                hit = json.loads(f.read_text())
                limit = EMPTY_TTL if hit.get("empty") else ttl
                if limit is None or time.time() - hit["t"] < limit:
                    return hit["data"]
            except Exception:
                pass
        data = self._fetch(url, allow_404, headers)
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({"t": time.time(), "data": data, "empty": data in ([], {}, None)}))
        except OSError:
            pass
        return data

    def _fetch(self, url, allow_404, headers):
        hdrs = {"User-Agent": UA, "Accept": "application/json", **self.headers, **(headers or {})}
        for attempt in range(4):
            self._wait()
            try:
                with _urlopen(urllib.request.Request(url, headers=hdrs), timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code == 404 and allow_404:
                    return []
                if e.code == 429 or e.code >= 500:
                    try:
                        retry = float(e.headers.get("Retry-After", 0)) if e.headers else 0.0
                    except (TypeError, ValueError):
                        retry = 0.0
                    time.sleep(min(max(retry, 2 ** (attempt + 1)), 60))
                    continue
                raise SourceError(f"{self.name}: HTTP {e.code} for {url}")
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
                if attempt == 3:
                    raise SourceError(f"{self.name}: {e}")
                time.sleep(2 ** attempt)
        raise SourceError(f"{self.name}: gave up on {url}")
