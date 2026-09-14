# SMART SURVEILLANCE SYSTEM

Hệ thống giám sát thông minh áp dụng các kỹ thuật Thị giác máy tính (Computer Vision) và Học sâu (Deep Learning) để theo dõi đối tượng, nhận diện chuyển động và phát hiện bất thường trong thời gian thực.

## Mục tiêu task
*   **Visualization:** Trực quan hoá toàn bộ kết quả lên video bằng OpenCV và giao diện Streamlit.

## Cấu trúc thư mục

\`\`\`text
smart-surveillance/
├── app/                  # Giao diện web demo (Streamlit)
├── data/
│   ├── input/            # Video đầu vào
│   └── output/           # Nơi Visualizer lưu kết quả tự động
├── models/               # Checkpoint/Trọng số mô hình (AutoEncoder, YOLOv8)
├── src/                  # Source code các tính năng lõi (Detection, Tracking, ReID...)
│   └── visualization/    # Module vẽ Bounding Box, Mask, Cảnh báo 
├── main.py               # File thực thi Pipeline
└── requirements.txt      # Các thư viện phụ thuộc
\`\`\`

## Hướng dẫn cài đặt và chạy thử nghiệm

**Bước 1: Cài đặt môi trường**
\`\`\`bash
# Khuyến nghị sử dụng Python 3.9+
pip install -r requirements.txt
\`\`\`

**Bước 2: Chạy hệ thống qua CLI (Pipeline)**
\`\`\`bash
python main.py --source data/input/test_video.mp4 --save
\`\`\`
*Kết quả sẽ tự động lưu vào thư mục `data/output/`.*

**Bước 3: Chạy giao diện Web (UI)**
\`\`\`bash
streamlit run app/streamlit_app.py
\`\`\`

## Bảng so sánh kết quả xử lý

| Tính năng | Phương pháp truyền thống | Phương pháp sử dụng (AI/DL) | Độ chính xác (Khung hình/giây) |
| :--- | :--- | :--- | :--- |
| **Detection** | HOG + SVM | YOLOv8 | Đạt ~ 45 FPS |
| **Tracking** | SORT | ByteTrack + Kalman Filter | Theo dõi mượt mà khi che khuất |
| **Anomaly** | Ngưỡng tĩnh | AutoEncoder (Deep Learning) | Tự thích ứng với bối cảnh mới |

## Demo Before - After (Visualization)


| Video Gốc (Test) | Video Kết Quả (Thành phẩm) |
| :---: | :---: |
| <video src="./data/test_videos/test.mp4" width="400" controls></video> | <video src="./outputs/smart_surveillance_output.mp4" width="400" controls></video> |

