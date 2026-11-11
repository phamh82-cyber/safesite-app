from tools.evaluate import score


def test_event_level_scoring():
    labels = {"a.mp4": [{"type": "FALL", "start": 5, "end": 6}],
              "b.mp4": [{"type": "NONE", "start": 0, "end": 0}]}
    preds = [{"video": "a.mp4", "type": "FALL", "t": 6.5},
             {"video": "a.mp4", "type": "FALL_IMMOBILE", "t": 8.0},   # gộp vào FALL, trong cửa sổ
             {"video": "b.mp4", "type": "FALL", "t": 3.0}]            # báo nhầm
    r = score(labels, preds)["FALL"]
    assert (r["TP"], r["FP"], r["FN"]) == (1, 1, 0)
    assert r["mean_latency_s"] == 1.5
