import cv2
import os

# === Cấu hình ===
video_path = 'source1.mp4'           # Đường dẫn đến video
output_dir = 'data'                  # Thư mục lưu frame
frame_rate = 5                       # Số frame mỗi giây muốn trích xuất (1 fps = mỗi giây lấy 1 frame)

# === Tạo thư mục lưu nếu chưa có ===
os.makedirs(output_dir, exist_ok=True)

# === Mở video ===
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)       # Frame per second gốc của video
frame_interval = int(fps / frame_rate)

frame_id = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % frame_interval == 0:
        filename = os.path.join(output_dir, f'frame_{saved_count:04d}.jpg')
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_id += 1

cap.release()
print(f"Đã lưu {saved_count} frames vào thư mục {output_dir}")
