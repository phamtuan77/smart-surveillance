## Module Visualization 

Module này chịu trách nhiệm nhận dữ liệu đầu ra từ các module khác (Detection, Tracking, Segmentation, Anomaly) và tổng hợp, vẽ trực quan lên video frame.

###  Chức năng chính:
*   Vẽ Bounding Box và gán ID của người/vật thể đang được theo dõi.
*   Hiển thị vùng chuyển động (Segmentation Mask) đè lên video với hiệu ứng trong suốt.
*   Hiển thị trạng thái cảnh báo trên màn hình: Bình thường (Màu xanh) hoặc Phát hiện bất thường (Màu đỏ) kèm theo Anomaly Score.