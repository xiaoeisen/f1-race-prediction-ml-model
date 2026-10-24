import pandas as pd
import pytest

pytest.importorskip("fastapi")


@pytest.fixture(scope="module")
def srv(raw):
    import server
    if not server.FILES["full"][0].exists():
        pytest.skip("models/ not trained yet (run: python train.py)")
    server.RATE_LIMIT = 0
    return server


@pytest.fixture
def client(srv):
    from fastapi.testclient import TestClient
    srv._hits.clear()
    return TestClient(srv.app)


def test_page_assets_and_security_headers(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY"
    assert client.get("/static/app.js").status_code == 200 and client.get("/static/style.css").status_code == 200
    assert client.get("/api/model").headers["cache-control"] == "no-store"


def test_health_reports_data_and_model(client, srv, monkeypatch):
    monkeypatch.setenv("AUTOPILOT", "0")
    j = client.get("/api/health").json()
    assert j["ok"] and j["data"]["rows"] > 1000 and j["model"]["version"] >= 1


def test_prediction_for_a_finished_race(client, srv, raw):
    y = int(raw.year.max())
    rnd = int(raw[raw.year == y]["round"].max()) - 1
    j = client.post("/api/predict", json={"year": y, "round": rnd}).json()
    assert j["meta"]["completed"] and j["meta"]["stage"] == "full"
    rows = j["rows"]
    assert 99 <= sum(r["win_%"] for r in rows) <= 101                        # win chances form a distribution
    assert [r["pred_pos"] for r in rows] == list(range(1, len(rows) + 1))
    assert all(r["podium_%"] <= 100 and r["podium_%"] >= r["win_%"] for r in rows)
    assert rows[0]["finish"] is not None                                      # the real result comes back alongside


def test_what_if_inputs_are_validated(client, raw):
    y = int(raw.year.max())
    rnd = int(raw[raw.year == y]["round"].max()) - 1
    ok = client.post("/api/predict", json={"year": y, "round": rnd, "grid": {raw.driver.iloc[0]: 20}, "rain": True})
    assert ok.status_code in (200, 400)                                       # 400 only if that driver wasn't in this race
    assert client.post("/api/predict", json={"year": y, "round": rnd, "grid": {"XXX": 3}}).status_code == 400
    assert client.post("/api/predict", json={"year": y, "round": rnd, "grid": {raw[raw.year == y].driver.iloc[0]: 99}}).status_code == 400
    assert client.post("/api/predict", json={"year": 1900, "round": 1}).status_code == 422
    assert client.post("/api/predict", json={"year": y, "round": 999}).status_code in (404, 422)


def test_upcoming_race_without_data_fails_politely(client, srv, monkeypatch, raw):
    import collect
    monkeypatch.setattr(collect, "collect_weekend", lambda *a, **k: pd.DataFrame())
    y = int(raw.year.max())
    last = int(raw[raw.year == y]["round"].max())
    r = client.post("/api/predict", json={"year": y, "round": last + 1})
    assert r.status_code in (404, 502) and "detail" in r.json()


def test_rate_limit_and_admin_token(client, srv, monkeypatch):
    monkeypatch.setattr(srv, "RATE_LIMIT", 3)
    codes = [client.get("/api/accuracy").status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
    assert client.get("/api/health").status_code in (200, 503)               # monitors are never rate limited
    monkeypatch.setattr(srv, "RATE_LIMIT", 0)
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    assert client.post("/api/autopilot/run").status_code == 403               # disabled unless a token is configured
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    assert client.post("/api/autopilot/run", headers={"x-admin-token": "wrong"}).status_code == 403
