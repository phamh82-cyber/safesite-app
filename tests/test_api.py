"""Kiểm thử API sửa camera (PUT /api/cameras/{id}). Không chạy mô hình: luồng camera
được thay bằng bộ ghi lại lệnh, cấu hình ghi vào thư mục tạm."""
import json

import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def client(tmp_path, monkeypatch):
    cams_file = tmp_path / "cameras.json"
    monkeypatch.setattr(main, "CAMERAS_PATH", cams_file)   # không đụng data/cameras.json thật
    monkeypatch.setattr(main, "cameras", {})
    monkeypatch.setattr(main, "workers", {})
    calls = []
    monkeypatch.setattr(main, "start_worker", lambda cid: calls.append(("start", cid)))
    monkeypatch.setattr(main, "restart_worker", lambda cid: calls.append(("restart", cid)))
    c = TestClient(main.app)
    c.calls, c.cams_file = calls, cams_file
    return c


def add_phone(client, src="http://10.188.157.135:8080/video"):
    r = client.post("/api/cameras", json={"name": "Điện thoại", "source_type": "rtsp", "source": src})
    assert r.status_code == 200
    return r.json()["id"]


def test_change_ip_keeps_id_and_calibration(client):
    cid = add_phone(client)
    cal = {"img_pts": [[0, 0], [10, 0], [10, 10], [0, 10]], "ground_pts": [[0, 0], [4, 0], [4, 6], [0, 6]]}
    assert client.put(f"/api/cameras/{cid}/calibration", json=cal).status_code == 200
    client.calls.clear()

    r = client.put(f"/api/cameras/{cid}", json={"source": "  http://10.46.209.220:8080/video "})
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == cid
    assert body["source"] == "http://10.46.209.220:8080/video"      # đã bỏ khoảng trắng
    assert body["calibration"] == cal
    assert client.calls == [("restart", cid)]                       # chạy lại với nguồn mới

    saved = json.loads(client.cams_file.read_text(encoding="utf-8"))
    assert saved[0]["source"] == "http://10.46.209.220:8080/video"
    assert saved[0]["calibration"] == cal


def test_rename_and_options_do_not_restart(client):
    cid = add_phone(client)
    client.calls.clear()
    r = client.put(f"/api/cameras/{cid}", json={"name": "Cổng kho A", "autostart": True})
    assert r.status_code == 200
    assert r.json()["name"] == "Cổng kho A" and r.json()["autostart"] is True
    assert r.json()["source"] == "http://10.188.157.135:8080/video"   # giữ nguyên trường không gửi
    assert client.calls == []


def test_same_source_does_not_restart(client):
    cid = add_phone(client)
    client.calls.clear()
    r = client.put(f"/api/cameras/{cid}", json={"source": "http://10.188.157.135:8080/video"})
    assert r.status_code == 200
    assert client.calls == []


def test_switch_to_webcam(client):
    cid = add_phone(client)
    r = client.put(f"/api/cameras/{cid}", json={"source_type": "webcam", "source": "0"})
    assert r.status_code == 200
    assert (r.json()["source_type"], r.json()["source"]) == ("webcam", "0")


@pytest.mark.parametrize("patch", [
    {"name": "   "},
    {"source": ""},
    {"source_type": "ftp"},
    {"source_type": "webcam", "source": "http://x/video"},
    {"source_type": "file", "source": "khong_ton_tai.mp4"},
])
def test_invalid_updates_rejected(client, patch):
    cid = add_phone(client)
    before = client.get("/api/cameras").json()[0]
    client.calls.clear()
    r = client.put(f"/api/cameras/{cid}", json=patch)
    assert r.status_code == 400
    after = client.get("/api/cameras").json()[0]
    assert {k: after[k] for k in ("name", "source_type", "source")} == \
           {k: before[k] for k in ("name", "source_type", "source")}  # không thay đổi gì
    assert client.calls == []


def test_unknown_camera_404(client):
    assert client.put("/api/cameras/nope", json={"name": "x"}).status_code == 404
