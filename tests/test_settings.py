"""Kiểm tra giá trị cài đặt, và API trả lỗi thay vì lưu giá trị sai."""
import copy
import json

import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import SETTINGS, Settings


@pytest.mark.parametrize("patch", [
    {"pipeline": {"process_fps": 0}},                     # chia cho 0 trong vòng lặp worker
    {"pipeline": {"process_fps": -5}},
    {"pipeline": {"process_fps": "abc"}},
    {"pipeline": {"process_fps": None}},
    {"pipeline": {"process_fps": float("nan")}},
    {"pipeline": {"imgsz": 640.5}},
    {"fall": {"lying_hold": -5}},
    {"fall": {"lying_angle": 200}},
    {"hazard": {"conf": 1.5}},
    {"hazard": {"every": 0}},
    {"objects": {"work_width": 10}},
    {"pipeline": {"pose_model": ""}},
    {"pipeline": {"pose_model": "evil.pkl"}},
    {"hazard": {"fire_model": "model.exe"}},
    {"hazard": {"enabled": "false"}},                     # phải là true/false thật
    {"pipeline": {"stabilize": 1}},
])
def test_invalid_values_rejected_and_nothing_applied(patch):
    s = Settings()
    before = copy.deepcopy(s.to_dict())
    good = {"fall": {"lying_hold": 2.0}}                  # giá trị hợp lệ đi kèm không được áp dụng
    errs = s.update({**good, **patch}, strict=True)
    assert errs and s.to_dict() == before


def test_valid_values_applied_with_coercion():
    s = Settings()
    assert s.update({"pipeline": {"process_fps": 12, "imgsz": 800.0, "stabilize": False},
                     "hazard": {"fire_model": " models/fire_smoke_best.pt "},
                     "height_fall": {"min_drop": 1.0}}, strict=True) == []
    assert s.pipeline.process_fps == 12.0 and s.pipeline.imgsz == 800 and s.pipeline.stabilize is False
    assert s.hazard.fire_model == "models/fire_smoke_best.pt" and s.height_fall.min_drop == 1.0


def test_loading_old_file_skips_only_bad_values():
    s = Settings()
    errs = s.update({"pipeline": {"process_fps": 0, "imgsz": 960}, "fall": {"lying_hold": 3}})
    assert len(errs) == 1 and s.pipeline.imgsz == 960 and s.fall.lying_hold == 3
    assert s.pipeline.process_fps == 10.0


def test_unknown_keys_ignored():
    s = Settings()
    assert s.update({"nope": {"x": 1}, "fall": {"khong_co": 1}}, strict=True) == []


# ----------------------------------------------------------------- API
@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "CAMERAS_PATH", tmp_path / "cameras.json")
    monkeypatch.setattr(main, "cameras", {})
    monkeypatch.setattr(main, "workers", {})
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")
    (tmp_path / "uploads").mkdir()
    monkeypatch.setattr(main, "save_settings", lambda: None)
    calls = []
    monkeypatch.setattr(main, "start_worker", lambda cid: calls.append(("start", cid)))
    monkeypatch.setattr(main, "restart_worker", lambda cid: calls.append(("restart", cid)))
    saved = copy.deepcopy(SETTINGS.to_dict())
    c = TestClient(main.app)
    c.calls, c.tmp = calls, tmp_path
    yield c
    SETTINGS.update(saved)


def post(c, **kw):
    body = {"name": "cam", "source_type": "rtsp", "source": "rtsp://10.0.0.5/s", **kw}
    return c.post("/api/cameras", json=body)


@pytest.mark.parametrize("body", [
    {"source_type": "ftp"},
    {"name": "  "},
    {"source": ""},
    {"source": "file:///etc/passwd"},                     # trước đây được nhận
    {"source": "javascript:alert(1)"},
    {"source_type": "webcam", "source": "rtsp://x"},
    {"source_type": "file", "source": "/etc/passwd"},     # file có thật nhưng ngoài thư mục upload
    {"source_type": "file", "source": "khong_co.mp4"},
])
def test_post_camera_validates(client, body):
    r = post(client, **body)
    assert r.status_code == 400, r.text
    assert client.get("/api/cameras").json() == [] and client.calls == []


def test_post_accepts_uploaded_file_and_valid_sources(client):
    f = client.tmp / "uploads" / "a.mp4"
    f.write_bytes(b"x")
    assert post(client, source_type="file", source=str(f)).status_code == 200
    assert post(client, source="http://10.0.0.7:8080/video").status_code == 200
    assert post(client, source_type="webcam", source="0").status_code == 200


def test_file_outside_upload_dir_rejected_via_put_too(client):
    cid = post(client).json()["id"]
    r = client.put(f"/api/cameras/{cid}", json={"source_type": "file", "source": "/etc/hostname"})
    assert r.status_code == 400


def test_rtsp_password_masked_but_kept(client):
    cid = post(client, source="rtsp://admin:s3cret@10.0.0.5:554/stream").json()["id"]
    assert "s3cret" not in client.get("/api/cameras").text
    assert client.get("/api/cameras").json()[0]["source"] == "rtsp://admin:***@10.0.0.5:554/stream"
    assert main.cameras[cid]["source"].endswith("admin:s3cret@10.0.0.5:554/stream")     # lưu bản gốc
    client.calls.clear()
    # form sửa gửi lại địa chỉ đã che ⇒ giữ mật khẩu cũ, không chạy lại camera
    r = client.put(f"/api/cameras/{cid}", json={"source": "rtsp://admin:***@10.0.0.5:554/stream"})
    assert r.status_code == 200 and client.calls == []
    assert main.cameras[cid]["source"] == "rtsp://admin:s3cret@10.0.0.5:554/stream"
    # đổi máy chủ nhưng giữ ***: mật khẩu cũ được áp cho địa chỉ mới, và camera chạy lại
    client.put(f"/api/cameras/{cid}", json={"source": "rtsp://admin:***@10.0.0.9:554/stream"})
    assert main.cameras[cid]["source"] == "rtsp://admin:s3cret@10.0.0.9:554/stream"
    assert client.calls == [("restart", cid)]
    assert "s3cret" not in json.dumps(client.put(f"/api/cameras/{cid}", json={"name": "x"}).json())


def test_put_settings_rejects_bad_values_without_changes(client):
    before = client.get("/api/settings").json()
    r = client.put("/api/settings", json={"pipeline": {"process_fps": 0}, "fall": {"lying_hold": 2.5}})
    assert r.status_code == 400 and "process_fps" in r.json()["detail"]
    assert client.get("/api/settings").json() == before


def test_put_settings_accepts_valid_and_new_sections(client):
    r = client.put("/api/settings", json={"hazard": {"min_hits": 4}, "objects": {"min_speed": 0.3},
                                           "height_fall": {"enabled": False}})
    assert r.status_code == 200
    s = client.get("/api/settings").json()
    assert s["hazard"]["min_hits"] == 4 and s["objects"]["min_speed"] == 0.3 and s["height_fall"]["enabled"] is False
