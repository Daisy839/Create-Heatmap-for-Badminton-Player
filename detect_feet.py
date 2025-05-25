import os
import cv2
import csv
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# ====== Khởi tạo model ======
model = YOLO('yolov8n.pt')  # Tải mô hình YOLOv8 nano

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# ====== Tọa độ 4 góc sân trong ảnh ======
court_points_image = np.array([
    [386, 314],  # top-left
    [892, 314],  # top-right
    [1046, 668], # bottom-right
    [226, 668]   # bottom-left
], dtype=np.float32)

# Tọa độ tương ứng trên sân thực (mét)
court_points_real = np.array([
    [0, 0],
    [6.1, 0],
    [6.1, 13.4],
    [0, 13.4]
], dtype=np.float32)

# Ma trận homography
H, _ = cv2.findHomography(court_points_image, court_points_real)

def to_real_world(x, y):
    pt = np.array([[[x, y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, H)
    return transformed[0][0]

def is_inside_court(x, y, width=6.1, height=13.4):
    return 0 <= x <= width and 0 <= y <= height

def extract_heels(image):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark
    h, w = image.shape[:2]

    points = [
        landmarks[mp_pose.PoseLandmark.LEFT_HEEL],
        landmarks[mp_pose.PoseLandmark.RIGHT_HEEL],
        landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX],
        landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
    ]

    xs = [pt.x * w for pt in points]
    ys = [pt.y * h for pt in points]

    return (sum(xs) / len(xs), sum(ys) / len(ys))

# ==== Đầu vào và đầu ra ====
input_folder = 'data'
output_csv = 'feet_world_coordinates.csv'
output_image_folder = 'detected_images'
os.makedirs(output_image_folder, exist_ok=True)  # Tạo thư mục nếu chưa có

with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['filename', 'player_id', 'image_x', 'image_y', 'real_x', 'real_y']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for filename in sorted(os.listdir(input_folder)):
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        path = os.path.join(input_folder, filename)
        img = cv2.imread(path)
        if img is None:
            continue

        print(f"🔄 Đang xử lý: {filename}")
        height, width = img.shape[:2]
        results = model(img)

        valid_players = []

        for det in results[0].boxes.data:
            cls_id = int(det[5])
            if cls_id != 0:
                continue  # Chỉ lấy người

            x1, y1, x2, y2 = map(int, det[:4])
            box_w, box_h = x2 - x1, y2 - y1
            if box_w * box_h < 0.01 * width * height:
                continue  # Bỏ người quá nhỏ

            cx, cy = (x1 + x2) // 2, y2  # Điểm dưới chân của bbox
            rx, ry = to_real_world(cx, cy)
            if is_inside_court(rx, ry):
                valid_players.append((x1, y1, x2, y2))

        # Chỉ lấy tối đa 2 người gần giữa sân nhất
        if len(valid_players) > 2:
            center_x = (court_points_image[0][0] + court_points_image[1][0]) / 2
            valid_players.sort(key=lambda b: abs((b[0] + b[2]) / 2 - center_x))
            valid_players = valid_players[:2]

        # Vẽ hộp giới hạn và xử lý tọa độ
        for idx, (x1, y1, x2, y2) in enumerate(valid_players):
            # Vẽ hộp giới hạn trên ảnh gốc
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'Person {idx}', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Trích xuất tọa độ mắt cá chân
            person_crop = img[y1:y2, x1:x2]
            feet = extract_heels(person_crop)
            if feet is None:
                continue

            avg_fx = feet[0] + x1
            avg_fy = feet[1] + y1

            # Chuyển đổi tọa độ đã điều chỉnh sang thực tế
            rx, ry = to_real_world(avg_fx, avg_fy)
            if is_inside_court(rx, ry):
                writer.writerow({
                    'filename': filename,
                    'player_id': idx,
                    'image_x': round(avg_fx, 2),
                    'image_y': round(avg_fy, 2),
                    'real_x': round(rx, 3),
                    'real_y': round(ry, 3)
                })

                # Vẽ điểm trung bình đã điều chỉnh
                cv2.circle(img, (int(avg_fx), int(avg_fy)), 5, (0, 0, 255), -1)
                cv2.putText(img, f'({int(avg_fx)}, {int(avg_fy)})', 
                            (int(avg_fx) + 10, int(avg_fy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Lưu ảnh đã detect
        output_path = os.path.join(output_image_folder, f'detected_{filename}')
        cv2.imwrite(output_path, img)

print(f"✅ Đã lưu dữ liệu tọa độ trung bình giữa hai chân vào: {output_csv}")
print(f"✅ Đã lưu các ảnh đã detect vào thư mục: {output_image_folder}")