📅 TUẦN 1 – NHIỆM VỤ CODE

Mục tiêu: Mỗi người làm xong phần code của mình, chạy thử độc lập. Tuần 2 mới ghép tất cả lại.

Người	Phần phụ trách	Nhiệm vụ đơn giản

1	🔍 Detection	Dùng YOLOv8 để phát hiện người và vật thể trong video.

2	👤 Tracking	Dùng ByteTrack để theo dõi người và gán ID cho từng người.

3	🔄 ReID	Nhận dạng lại người khi người đó xuất hiện lại trong video.

4	🟦 Segmentation	Phát hiện và đánh dấu vùng đang chuyển động trong video.

5	🚨 Anomaly	Tính Anomaly Score và xác định chuyển động bình thường / bất thường.

6	🎨 Visualization	Vẽ khung người, ID, vùng chuyển động và cảnh báo lên video.

7	🔗 Pipeline	Tạo phần kết nối để chuẩn bị ghép các module thành một chương trình.
Chi tiết dễ hiểu hơn

👤 Người 1 – Detection

Video → YOLOv8 → Phát hiện người/vật thể

👤 Người 2 – Tracking

Người → ByteTrack → ID 01, ID 02, ID 03...

👤 Người 3 – ReID

Người xuất hiện lại → ReID → Kiểm tra có phải người cũ không

👤 Người 4 – Segmentation

Video → Motion Detection → Vùng chuyển động

👤 Người 5 – Anomaly

Chuyển động → Phân tích → Normal / Abnormal

👤 Người 6 – Visualization

Kết quả → Vẽ lên video → Box + ID + cảnh báo

👤 Người 7 – Pipeline

Chuẩn bị kết nối:
Detection → Tracking → ReID → Segmentation → Anomaly → Visualization

LƯU Ý
Mỗi người làm phần riêng, không sửa trực tiếp main cùng lúc.
Làm lưu phần test của mình trong test 
