import cv2
import numpy as np
from feature_extractor import FeatureExtractor
from reid_model import ReIDModel

extractor = FeatureExtractor(device="cpu")
reid = ReIDModel(threshold=0.7)

# Đọc ảnh thật
img1 = cv2.imread("test_images/person1_a.jpg")
img2 = cv2.imread("test_images/person1_b.jpg")
img3 = cv2.imread("test_images/person2.jpg")

if img1 is None or img2 is None or img3 is None:
    print("Lỗi: không đọc được 1 trong 3 ảnh. Kiểm tra lại tên file/đường dẫn.")
    exit()

# Trích đặc trưng
feat1 = extractor.extract(img1)
feat2 = extractor.extract(img2)
feat3 = extractor.extract(img3)

reid.add_to_gallery("person_01", feat1)

matched_id, score = reid.match(feat2)
print(f"Test 1 (cung nguoi): matched_id={matched_id}, score={score:.3f}")

matched_id2, score2 = reid.match(feat3)
print(f"Test 2 (nguoi khac): matched_id={matched_id2}, score={score2:.3f}")

# ==== TẠO ẢNH BẰNG CHỨNG ====
def resize_h(img, h=250):
    ratio = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1]*ratio), h))

img1_r = resize_h(img1)
img2_r = resize_h(img2)
img3_r = resize_h(img3)

# Ghép ảnh 1 (gallery) và ảnh 2 (test match) cạnh nhau
row1 = np.hstack([img1_r, img2_r])
label1 = f"Test1: {matched_id} (score={score:.3f})"
canvas1 = np.zeros((row1.shape[0]+50, row1.shape[1], 3), dtype=np.uint8)
canvas1[50:, :, :] = row1
cv2.putText(canvas1, label1, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

# Ghép ảnh 1 (gallery) và ảnh 3 (người khác) cạnh nhau
row2 = np.hstack([img1_r, img3_r])
label2 = f"Test2: {matched_id2} (score={score2:.3f})"
canvas2 = np.zeros((row2.shape[0]+50, row2.shape[1], 3), dtype=np.uint8)
canvas2[50:, :, :] = row2
cv2.putText(canvas2, label2, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

# Ghép 2 kết quả lại thành 1 ảnh duy nhất
max_w = max(canvas1.shape[1], canvas2.shape[1])
def pad_w(c, w):
    pad = np.zeros((c.shape[0], w-c.shape[1], 3), dtype=np.uint8)
    return np.hstack([c, pad])
canvas1 = pad_w(canvas1, max_w)
canvas2 = pad_w(canvas2, max_w)

final = np.vstack([canvas1, canvas2])
cv2.imwrite("reid_test_result.jpg", final)
print("Da luu anh bang chung: reid_test_result.jpg")