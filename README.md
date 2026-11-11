# SafeSite – Hệ thống thông minh phát hiện tai nạn lao động

Web dashboard phát hiện **té ngã**, **nguy cơ va chạm / va chạm người – phương tiện** và **va chạm phương tiện – phương tiện** từ video giám sát.
Môn Phát triển hệ thống thông minh – Nhóm: Nguyễn Đức Thuận, Nguyễn Đại Nghĩa, Nguyễn Minh Hiển.

## 1. Cài đặt

Yêu cầu Python 3.10+ (đã chạy thử với 3.13). Có GPU NVIDIA thì nhanh hơn, CPU vẫn chạy được (~5–10 FPS với mô hình `n`).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Máy chỉ có CPU, cài PyTorch bản CPU trước để nhẹ hơn nhiều:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

Lần chạy đầu, Ultralytics tự tải `yolo11n-pose.pt` và `yolo11n.pt` (~6 MB mỗi file).

## 2. Chạy dashboard

```bash
python run.py            # mở http://localhost:8000
```

1. Vào **Camera** → chọn *File video* / *Webcam* (nhập `0`) / *Camera IP* (`rtsp://...` hoặc `http://...`) → **Thêm và chạy**. Dùng điện thoại làm camera: xem [mục 3](#3-dùng-điện-thoại-làm-camera). Đổi tên, địa chỉ hoặc tùy chọn của camera đã thêm: bấm **Sửa** (giữ nguyên hiệu chỉnh).
2. (Nên làm) bấm **Hiệu chỉnh**: chọn 4 góc một hình chữ nhật trên nền, nhập kích thước thật (m). Không hiệu chỉnh thì hệ thống tự ước lượng tỉ lệ từ chiều cao người/xe (kém chính xác hơn, có ghi trên giao diện).
3. Tab **Giám sát** hiển thị luồng đã phân tích và cảnh báo realtime (âm thanh + thông báo trình duyệt).
4. Tab **Sự kiện**: xem ảnh + clip 6 giây, đánh dấu *Xác nhận / Đã xử lý / Báo nhầm*, xuất CSV.
5. Tab **Cài đặt**: chỉnh ngưỡng, có hiệu lực ngay.

## 3. Dùng điện thoại làm camera

Điện thoại chạy app phát video qua HTTP, SafeSite đọc như một camera IP. Không cần sửa code.

1. Cài app: **IP Webcam** (Android, cổng `8080`) hoặc **DroidCam** (Android/iPhone, cổng `4747`).
2. Trong IP Webcam → *Video preferences* → **Video resolution: 640×480**. Mô hình chỉ dùng ảnh 640 px nên độ phân giải cao hơn chỉ làm chậm; trên một số máy (đã gặp với Redmi) để 1920×1080 thì camera không phát khung nào, màn hình xem trước đen.
3. Bấm **Start server**, ghi lại địa chỉ app hiện, VD `http://192.168.1.25:8080`.
4. Kiểm tra trên trình duyệt của máy tính: mở `http://192.168.1.25:8080/shot.jpg` phải thấy ảnh.
5. Dashboard → **Camera** → *Camera IP (RTSP/HTTP)* → địa chỉ **có đuôi `/video`**:
   - IP Webcam: `http://192.168.1.25:8080/video`
   - DroidCam: `http://192.168.1.25:4747/video`

| Hiện tượng | Nguyên nhân | Cách xử lý |
|---|---|---|
| *Không mở được nguồn video*, ping tới điện thoại thất bại | Hai máy khác mạng, hoặc Wi‑Fi bật chặn thiết bị nói chuyện với nhau (AP isolation – hay gặp ở Wi‑Fi nhà trọ, quán, trường) | Bật **hotspot trên điện thoại**, cho laptop vào hotspot đó; mở lại app để lấy IP mới. Video đi trong mạng nội bộ, không tốn 4G |
| Mở được trang `http://IP:8080/` nhưng `/shot.jpg` và `/video` treo | Camera không phát khung: độ phân giải không hỗ trợ, app khác đang chiếm camera, app bị hệ điều hành tắt camera khi chạy nền | Giảm độ phân giải xuống 640×480; để app ở màn hình chính; tắt app camera khác; cho app *Không hạn chế pin* |
| Xem trước trong app màu đen | Quyền camera bị chặn (công tắc *Truy cập camera* ở Trung tâm điều khiển, hoặc quyền của app) | Bật lại quyền, khởi động lại điện thoại |
| Nhập `http://IP:8080/` không chạy | Đó là trang web của app, không phải luồng video | Thêm đuôi `/video` |
| Camera đang chạy rồi chuyển *Đang kết nối lại* | Điện thoại rớt mạng / app dừng | Tự chạy tiếp khi có hình lại. Nếu IP điện thoại đã đổi (bật lại hotspot), bấm **Sửa** và nhập IP mới |

Đặt điện thoại cố định, cao ~1–1,5 m, nhìn chéo xuống; cắm sạc. Camera rung làm sai vận tốc → báo nhầm.

### Độ trễ với camera trực tiếp

Camera phát ~30 khung/giây nhưng YOLO trên CPU chỉ xử lý ~6 khung/giây. Nếu đọc khung tuần tự, các khung chưa xử lý xếp hàng trong bộ đệm của OpenCV/FFmpeg và hình bị **trễ dồn**: càng chạy lâu càng chậm, cảnh báo cũng đến muộn theo.

Vì vậy webcam / camera IP được đọc qua `LiveReader` ([app/pipeline.py](app/pipeline.py)): một luồng riêng đọc liên tục và **chỉ giữ khung mới nhất**, vòng phân tích luôn lấy khung hiện tại (khung cũ bị bỏ). Ngoài ra:

- thời điểm `t` của mỗi khung là lúc **nhận được khung**, không phải lúc xử lý xong → vận tốc, TTC chính xác hơn;
- camera mạng mất hình > 3 s (kể cả lúc bấm Chạy mà điện thoại chưa phát hình) thì chuyển *Đang kết nối lại* và tự mở lại mỗi giây, có hình là chạy tiếp; webcam USB không mở được hoặc mất hình thì báo lỗi;
- mở/đọc camera mạng có timeout 5 s (mặc định của FFmpeg là 30 s).

File video vẫn đọc tuần tự (không bỏ khung) để kết quả đánh giá lặp lại được.

Đo trên điện thoại thật (IP Webcam 640×480, ~31 FPS, giả lập 150 ms xử lý/khung, chạy 15 s):

| Cách đọc | Độ trễ hình so với thực tế |
|---|---|
| Tuần tự (cũ) | ≈ 11,7 s và tăng dần |
| `LiveReader` | ≈ 17 ms (tối đa 38 ms) + thời gian xử lý một khung |

## 4. Các loại sự kiện

| Mã | Mức | Điều kiện chính |
|---|---|---|
| `FALL` | Nghiêm trọng | Rơi nhanh (hông ≥ 0,8 chiều cao/giây hoặc chiều cao giảm ≥ 35% trong 0,7 s) → nằm (góc thân ≥ 60°) trong ≤ 2 s → giữ ≥ 1 s |
| `FALL_IMMOBILE` | Nghiêm trọng | Sau FALL, bất động > 10 s |
| `PERSON_DOWN` | Cảnh báo | Nằm bất động ≥ 5 s nhưng không thấy pha rơi (ngã từ từ / mất pha rơi) |
| `NEAR_MISS` | Cảnh báo | Xe đang chạy cách người < 2 m, hoặc TTC < 2 s |
| `PERSON_VEHICLE_COLLISION` | Nghiêm trọng | Tiếp xúc (< 0,4 m trên nền + bbox giao nhau) **và** trong 2 s người ngã / bị đẩy / mất dấu; hoặc FALL ≤ 3 s sau khi ở sát xe đang chạy |
| `VEHICLE_COLLISION` | Nghiêm trọng | Footprint hai xe chạm nhau, trước đó tiến lại gần > 1 m/s, sau đó một xe giảm tốc > 50% hoặc đổi hướng > 30° |
| `FALL_FROM_HEIGHT` | Nghiêm trọng | Hông rơi ≥ 0,7 chiều cao người, tốc độ ≥ 2 H/s, đang tăng tốc, bbox không to lên (loại chạy lại gần camera) → sau đó người nằm, hoặc mất dấu (kể cả rơi ra khỏi khung). Người hạ cánh đứng và đi tiếp thì không báo. Kèm độ cao rơi ước tính (tham khảo, chuẩn 1,7 m) |
| `OBJECT_FALL` | Cảnh báo | Vệt chuyển động (trừ nền MOG2 trên ảnh đã bù rung) đi xuống ≥ 12% chiều cao khung, chủ yếu theo phương đứng, đang tăng tốc, tốc độ cuối ≥ 0,25 khung/s, không phải người/xe |
| `STRUCK_BY` | Nghiêm trọng | Vật rơi (như trên) có tâm lọt vào bbox người (mở rộng 15%) **và** trong 2 s người ngã hoặc mất dấu. Vật trúng mà người vẫn đứng ⇒ chỉ có `OBJECT_FALL` |
| `FIRE_SMOKE` | Nghiêm trọng (lửa) / Cảnh báo (khói) | ≥ 3 lần phát hiện cùng vị trí trong 3 s. Nguồn: mô hình YOLO fine-tune (mục 8) hoặc, nếu chưa có, bộ lọc màu lửa + yêu cầu vùng màu nhấp nháy (chỉ nhận ra lửa, dễ báo nhầm hơn) |

**Bù rung camera** (`app/ego.py`): mỗi khung ước lượng dịch chuyển toàn cảnh bằng optical flow trên nền (bỏ vùng người/xe). Tọa độ trên nền và ảnh cho phân tích vật rơi được ổn định theo đó; khi rung mạnh (> 0,6% chiều rộng ảnh/khung) thì tạm dừng phân tích vật rơi và bộ lọc màu lửa. Tắt/bật ở *Cài đặt → Xử lý video → Bù rung camera*.

## 5. Kiểm thử và đánh giá

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests                     # ~116 kiểm thử, không cần GPU/video/camera (~1 phút)
python -m tools.run_video video.mp4 --out out.mp4
python -m tools.evaluate --labels data/labels.csv --videos data/test_videos --out data/eval
```

| File | Nội dung |
|---|---|
| `tests/test_scenarios.py` | 16 kịch bản té ngã / va chạm trên dữ liệu giả lập |
| `tests/test_evaluate.py` | chấm điểm theo sự kiện |
| `tests/test_live.py` | `LiveReader` với camera giả: luôn trả khung mới nhất, không trả lặp, timeout khi camera treo, tự kết nối lại camera mạng (cả khi lần mở đầu thất bại), webcam không mở được / mất hình thì báo lỗi, dừng luồng đọc |
| `tests/test_api.py` | API sửa camera: đổi IP giữ ID và hiệu chỉnh, chỉ chạy lại khi nguồn đổi, từ chối dữ liệu sai (tên trống, loại nguồn lạ, webcam không phải số, file không tồn tại) |

`labels.csv` (một dòng cho mỗi sự kiện; video không có sự kiện ghi `NONE`):

```csv
video,type,start,end
fall_01.mp4,FALL,4.2,5.6
kneel_03.mp4,NONE,0,0
cart_02.mp4,NEAR_MISS,7.0,8.5
```

Tạo nhanh video bán giả lập (cắt người thật đang đứng, xoay quanh bàn chân để mô phỏng ngã) khi chưa có dữ liệu – chỉ để kiểm tra luồng xử lý, không thay thế dữ liệu thật:

```bash
python -m tools.make_semisynthetic src.mp4 --frame 50 --box 472,80,556,260 --bg 100 --out fall_synth.mp4
```

Đánh giá theo **sự kiện**: cảnh báo cùng loại trong `[start − 1 s, end + 3 s]` là TP. Báo cáo Precision, Recall, F1, độ trễ trung bình, FP/giờ.
Để làm thí nghiệm so sánh A–D, chạy lại với các cấu hình khác nhau và lưu `report.json` của từng lần.

## 6. Cấu trúc mã

```
app/
  config.py        ngưỡng mặc định (dataclass) – lưu ở data/settings.json
  tracks.py        Track, lịch sử, vận tốc trên nền; nối lại ID
  geometry.py      homography / tự hiệu chỉnh từ chiều cao vật, đa giác footprint
  analyzers/fall.py       máy trạng thái té ngã trên pose
  analyzers/collision.py  người–xe, xe–xe, TTC, near-miss, xác nhận va chạm
  analyzers/height_fall.py  ngã từ trên cao (rơi gần tự do trên track người)
  analyzers/objects.py      vật rơi / bị vật đập trúng (trừ nền MOG2 + ghép vệt)
  analyzers/hazard.py       cháy/khói: xác nhận theo thời gian, bộ lọc màu dự phòng
  ego.py           bù rung camera (optical flow)
  engine.py        lõi phân tích 1 camera (không phụ thuộc mô hình → test được)
  pipeline.py      đọc video (LiveReader cho camera trực tiếp), YOLO + ByteTrack, luồng xử lý mỗi camera
  annotate.py      vẽ khung xương, bbox, khoảng cách, bản đồ mặt nền
  events.py        SQLite, ảnh chụp, clip .webm, WebSocket hub
  main.py          FastAPI: REST, WebSocket, MJPEG
static/            giao diện (HTML/CSS/JS thuần, không cần build)
tools/             evaluate.py, run_video.py
kaggle/            train_fire_smoke.ipynb – huấn luyện mô hình cháy/khói trên Kaggle (GPU)
models/            đặt file .pt đã huấn luyện (không đưa lên git)
tests/             bộ giả lập + kịch bản kiểm thử
```

## 7. Nhận diện xe nâng, máy xúc (fine-tune)

COCO **không có** lớp forklift/excavator; bản mặc định dùng car/truck/bus/motorcycle/bicycle. Để dùng trong kho/công trường:

1. Đặt ảnh và nhãn (định dạng YOLO) vào `data/construction/` theo cấu trúc ghi trong [construction.yaml](construction.yaml): `images/train`, `images/val`, `labels/train`, `labels/val`. Chỉ gán nhãn phương tiện, người đã có mô hình pose lo. Có thể gán nhãn bằng Roboflow, CVAT hoặc Label Studio rồi xuất ra định dạng *YOLO*. Thư mục này không được commit lên git.
2. Sửa danh sách `names` trong `construction.yaml` nếu dùng bộ lớp khác (mặc định 11 lớp: forklift, excavator, wheel_loader, …).
3. Huấn luyện:

```bash
yolo detect train model=yolo11n.pt data=construction.yaml epochs=50 imgsz=640
```

Sau đó đặt đường dẫn `runs/detect/train/weights/best.pt` vào *Cài đặt → Mô hình phát hiện xe* và sửa `VEHICLE_CLASSES`, `CLASS_HEIGHT_M` trong `app/config.py` theo chỉ số và tên lớp mới. Ví dụ với bộ lớp mặc định: `VEHICLE_CLASSES = {0: "forklift", 1: "excavator", ..., 10: "truck"}`.

## 8. Huấn luyện mô hình cháy/khói (Kaggle)

1. Trên Kaggle: *Create → New Notebook → File → Import Notebook* → chọn `kaggle/train_fire_smoke.ipynb`.
2. *Settings → Accelerator*: GPU (T4). *Add Input*: thêm một dataset YOLO về lửa/khói (ví dụ D-Fire; **tự kiểm tra tên dataset, giấy phép và thứ tự lớp** trước khi dùng).
3. Chạy lần lượt các ô. Lần đầu nên đặt `EPOCHS = 3` để chắc pipeline chạy, rồi mới tăng lên.
4. Tải `fire_smoke_best.pt` ở tab *Output*, đặt vào `models/`.
5. Dashboard → *Cài đặt → Cháy / khói → Mô hình* = `models/fire_smoke_best.pt` → lưu → chạy lại camera. Log sẽ cảnh báo nếu không thấy file hoặc mô hình không có lớp fire/smoke.

Tên lớp được nhận theo từ khóa `fire`/`flame`/`smoke`; lớp khác bị bỏ qua.

## 9. Giới hạn đã biết

- **Các loại sự kiện thêm sau (`FALL_FROM_HEIGHT`, `OBJECT_FALL`, `STRUCK_BY`, `FIRE_SMOKE`) mới được kiểm thử trên mô phỏng/video tổng hợp, chưa đo trên video thật.** Cần chạy `tools/evaluate.py` trên clip thật và đo báo nhầm/giờ trước khi dựa vào.
- Ngã từ trên cao: cần camera nhìn ngang hoặc chéo (nhìn thẳng từ trên xuống mất tín hiệu rơi theo phương đứng); độ cao rơi ước tính chịu sai số phối cảnh; người hạ cánh không nằm và không mất dấu (ngồi sụp, bò) sẽ không được báo.
- Trong các ngưỡng ngã cao, `min_speed` và `min_drop` được test mô phỏng chứng minh vai trò; `min_accel` và `max_growth` là lớp phòng thêm chưa được chứng minh riêng.
- Vật rơi: vật nhỏ hơn ~0,03% khung, cùng màu nền, hoặc rơi sau lưng người (bbox trùng nhưng ở xa) có thể bị bỏ sót hoặc tính nhầm; mưa/tuyết dày bị lọc bằng diện tích và số vệt tối đa. Vật trúng mà người không phản ứng không tạo `STRUCK_BY`.
- Cháy/khói: bộ lọc màu chỉ nhận lửa và dễ báo nhầm (đèn hàn, đèn pha, hoàng hôn); khói cần mô hình. Điện giật, nổ, sập đổ, kẹt/kẹp **chưa được hỗ trợ**.
- Bù rung chỉ xử lý tịnh tiến; xoay/zoom mạnh hoặc cảnh thiếu vân (ban đêm) thì không đo được và không bù.
- API chưa có xác thực: chỉ chạy ở `127.0.0.1` hoặc mạng tin cậy.


- Mô hình COCO nhận diện kém xe nhìn thẳng từ trên xuống (camera trần, drone): khi thử, ô tô nhìn từ trên bị nhận thành “cell phone”. Cần camera nhìn chéo hoặc fine-tune.
- Footprint xe xấp xỉ bằng 35% phần dưới bbox: đúng nhất khi camera gắn cao, nhìn chéo xuống.
- Homography giả định nền phẳng; dốc, bậc thang, giàn giáo sẽ đo sai.
- Ngưỡng mặc định chưa được hiệu chỉnh trên dữ liệu thật – cần chạy `tools/evaluate.py` trên bộ kiểm thử của nhóm.
- Hệ thống chỉ hỗ trợ người giám sát, không thay thế quy trình an toàn lao động.
