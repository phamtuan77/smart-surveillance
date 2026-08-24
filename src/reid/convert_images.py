from PIL import Image
import os

folder = "test_images"

# Sửa lại đúng tên file gốc bạn đang có trong thư mục test_images
files = os.listdir(folder)
print("Các file hiện có:", files)

for f in files:
    if f.lower().endswith((".webp", ".png", ".jfif", ".bmp")):
        path = os.path.join(folder, f)
        img = Image.open(path).convert("RGB")
        new_name = os.path.splitext(f)[0] + ".jpg"
        new_path = os.path.join(folder, new_name)
        img.save(new_path, "JPEG")
        print(f"Da chuyen: {f} -> {new_name}")