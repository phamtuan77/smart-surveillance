# SMART SURVEILLANCE

## Cấu trúc Project -- Phát hiện & phân vùng hành vi bất thường trong giám sát

### 1. Mục tiêu Project

Project xây dựng hệ thống Camera an ninh thông minh (Smart Surveillance)
có khả năng:

-   Phát hiện người và vật thể (Detection)
-   Theo dõi người qua nhiều khung hình (Tracking)
-   Nhận dạng lại người (Person Re-Identification / ReID)
-   Xác định vùng chuyển động (Motion Segmentation)
-   Phát hiện hành vi/chuyển động bất thường (Anomaly Detection)
-   Hiển thị kết quả và cảnh báo (Visualization & Alert)

------------------------------------------------------------------------

## 2. Ngôn ngữ và công nghệ

  Thành phần     Công nghệ              Mục đích
  -------------- ---------------------- ----------------------------------
  Ngôn ngữ       Python                 Ngôn ngữ chính của Project
  Xử lý video    OpenCV                 Đọc camera, video và xử lý frame
  Detection      YOLOv8                 Phát hiện người/vật thể
  Tracking       ByteTrack              Theo dõi người và giữ ID
  ReID           PyTorch + ReID Model   Nhận dạng lại người
  Segmentation   OpenCV / PyTorch       Xác định vùng chuyển động
  Anomaly        AutoEncoder            Phát hiện bất thường
  Giao diện      Streamlit              Hiển thị kết quả demo

------------------------------------------------------------------------

## 3. Cấu trúc thư mục

``` text
smart-surveillance/
│
├── data/
│   ├── raw/
│   │   ├── ucsd_ped1/
│   │   ├── ucsd_ped2/
│   │   └── avenue/
│   ├── processed/
│   └── test_videos/
│
├── models/
│   ├── detection/
│   ├── tracking/
│   ├── reid/
│   └── anomaly/
│
├── src/
│   ├── detection/
│   │   ├── detector.py
│   │   └── config.py
│   │
│   ├── tracking/
│   │   ├── tracker.py
│   │   └── trajectory.py
│   │
│   ├── reid/
│   │   ├── reid_model.py
│   │   └── feature_extractor.py
│   │
│   ├── segmentation/
│   │   ├── motion.py
│   │   └── mask.py
│   │
│   ├── anomaly/
│   │   ├── autoencoder.py
│   │   └── detector.py
│   │
│   └── visualization/
│       ├── draw_boxes.py
│       ├── draw_tracks.py
│       └── draw_alert.py
│
├── app/
│   └── app.py
│
├── notebooks/
│   ├── 01_detection.ipynb
│   ├── 02_tracking.ipynb
│   ├── 03_reid.ipynb
│   └── 04_anomaly.ipynb
│
├── tests/
│   ├── test_detection.py
│   ├── test_tracking.py
│   └── test_anomaly.py
│
├── outputs/
│   ├── detection/
│   ├── tracking/
│   ├── anomaly/
│   └── videos/
│
├── main.py
├── pipeline.py
├── config.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

------------------------------------------------------------------------

## 4. Ý nghĩa từng thư mục

### `data/` -- Dữ liệu

Chứa dataset và video dùng để thử nghiệm.

-   `raw/`: dữ liệu gốc.
-   `processed/`: dữ liệu đã xử lý.
-   `test_videos/`: video dùng để demo.

**Lưu ý:** Dataset lớn không nên đưa trực tiếp lên GitHub.

------------------------------------------------------------------------

### `models/` -- Model

Chứa model hoặc file trọng số.

-   `detection/`: YOLOv8.
-   `tracking/`: model/cấu hình tracking nếu cần.
-   `reid/`: ReID model.
-   `anomaly/`: AutoEncoder.

**Lưu ý:** File model lớn nên để ngoài GitHub hoặc dùng Git LFS nếu cần.

------------------------------------------------------------------------

### `src/detection/` -- Phát hiện

Phụ trách:

> YOLOv8 → phát hiện người/vật thể.

Ví dụ:

``` text
Video Frame
    ↓
YOLOv8
    ↓
Person + Bounding Box + Confidence
```

Người phụ trách: **Thành viên 1**

------------------------------------------------------------------------

### `src/tracking/` -- Theo dõi

Phụ trách:

> Theo dõi người và gán ID.

Ví dụ:

``` text
Person
  ↓
ByteTrack
  ↓
ID 01 → ID 01 → ID 01
```

`trajectory.py` dùng để lưu và vẽ đường đi của người.

Người phụ trách: **Thành viên 3**

------------------------------------------------------------------------

### `src/reid/` -- Nhận dạng lại người

Phụ trách:

> Kiểm tra người xuất hiện lại có phải người cũ hay không.

Ví dụ:

``` text
ID 01
 ↓
Bị che khuất
 ↓
Xuất hiện lại
 ↓
ReID
 ↓
ID 01
```

Người phụ trách: **Thành viên 4**

------------------------------------------------------------------------

### `src/segmentation/` -- Phân vùng chuyển động

Phụ trách:

> Tìm vùng đang chuyển động trong video.

Giai đoạn đầu có thể dùng:

-   OpenCV
-   MOG2
-   Optical Flow

Ví dụ:

``` text
Video
 ↓
Motion Detection
 ↓
Motion Mask
```

Người phụ trách: **Thành viên 5**

------------------------------------------------------------------------

### `src/anomaly/` -- Phát hiện bất thường

Phụ trách:

> Phân biệt chuyển động bình thường và bất thường.

Mô hình dự kiến:

**AutoEncoder**

Luồng:

``` text
Video
 ↓
AutoEncoder
 ↓
Reconstruction Error
 ↓
Anomaly Score
 ↓
Normal / Abnormal
```

Người phụ trách: **Thành viên 6**

------------------------------------------------------------------------

### `src/visualization/` -- Hiển thị

Phụ trách:

-   Vẽ Bounding Box.
-   Hiển thị ID.
-   Vẽ trajectory.
-   Hiển thị vùng bất thường.
-   Hiển thị cảnh báo.

Người phụ trách: **Thành viên 8**

------------------------------------------------------------------------

### `app/` -- Giao diện

Chứa giao diện demo bằng Streamlit.

Giao diện có thể hiển thị:

``` text
SMART SURVEILLANCE

Video
 ├── Person ID 01
 ├── Person ID 02
 └── Person ID 03

Person Count: 3
Tracking: Active
Anomaly Score: 0.82
Status: ABNORMAL
```

Người phụ trách: **Thành viên 8**

------------------------------------------------------------------------

### `tests/` -- Kiểm thử

Dùng để kiểm tra từng phần:

-   Detection có chạy không?
-   Tracking có giữ đúng ID không?
-   Anomaly có trả kết quả không?

Người phụ trách: **Thành viên 7**

------------------------------------------------------------------------

### `outputs/` -- Kết quả

Lưu:

-   Kết quả Detection.
-   Kết quả Tracking.
-   Kết quả Anomaly.
-   Video sau khi xử lý.

------------------------------------------------------------------------

## 5. Luồng hoạt động chính

Đây là phần quan trọng nhất của Project:

``` text
                 CAMERA / VIDEO
                       │
                       ▼
                  YOLOv8
                 Detection
                       │
                       ▼
                  ByteTrack
                  Tracking
                       │
                       ▼
                    ReID
             Person Identity
                       │
                       ▼
             Motion Segmentation
                       │
                       ▼
             Anomaly Detection
                 AutoEncoder
                       │
                       ▼
                Normal / Abnormal
                       │
                       ▼
               Alert + Visualization
```

------------------------------------------------------------------------

## 6. `main.py` và `pipeline.py`

### `main.py`

Là file dùng để chạy chương trình.

Ví dụ:

``` text
python main.py
```

### `pipeline.py`

Là file kết nối các phần với nhau:

``` text
Detection
    ↓
Tracking
    ↓
ReID
    ↓
Segmentation
    ↓
Anomaly Detection
    ↓
Visualization
```

Không nên để toàn bộ code trong `main.py`.

------------------------------------------------------------------------

## 7. Chia việc cho 8 thành viên

  Thành viên   Phần phụ trách                   Folder chính
  ------------ -------------------------------- -------------------------------
  Người 1      YOLOv8 Detection                 `src/detection/`
  Người 2      Dataset + Detection Evaluation   `data/` + evaluation
  Người 3      ByteTrack Tracking               `src/tracking/`
  Người 4      Person ReID                      `src/reid/`
  Người 5      Motion Segmentation              `src/segmentation/`
  Người 6      AutoEncoder Anomaly              `src/anomaly/`
  Người 7      Testing + Metrics                `tests/`
  Người 8      UI + Integration                 `app/` + `src/visualization/`

------------------------------------------------------------------------

## 8. Quy tắc làm GitHub

Mỗi người làm phần riêng, không sửa trực tiếp `main` cùng lúc.

``` text
main
│
├── feature/detection
├── feature/dataset
├── feature/tracking
├── feature/reid
├── feature/segmentation
├── feature/anomaly
├── feature/evaluation
└── feature/ui
```

Quy trình:

``` text
Pull code mới
    ↓
Tạo branch riêng
    ↓
Code
    ↓
Commit
    ↓
Push
    ↓
Pull Request
    ↓
Review
    ↓
Merge vào main
```

