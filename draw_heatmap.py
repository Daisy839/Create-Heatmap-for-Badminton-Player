import csv
import numpy as np
import cv2

# Kích thước sân chuẩn: 6.1m x 13.4m
width_m, height_m = 6.1, 13.4

# Kích thước ảnh: 610 x 1340 pixels
width_px, height_px = 610, 1340

# Scale chuyển đổi mét -> pixel
scale_x = width_px / width_m
scale_y = height_px / height_m

def to_pixel(x, y):
    return int(x * scale_x), int(y * scale_y)

# Tạo kernel Gaussian 2D để mô phỏng bóng đổ
def create_gaussian_kernel(size=30, sigma=10):
    return cv2.getGaussianKernel(size, sigma) @ cv2.getGaussianKernel(size, sigma).T

gaussian_kernel = create_gaussian_kernel()

# Đọc dữ liệu
coords = []
with open('feet_world_coordinates.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        x, y = float(row['real_x']), float(row['real_y'])
        coords.append(to_pixel(x, y))

# Tạo heatmap
heatmap = np.zeros((height_px, width_px), dtype=np.float32)
kernel_size = gaussian_kernel.shape[0]
offset = kernel_size // 2

for x, y in coords:
    if 0 <= x < width_px and 0 <= y < height_px:
        # Áp dụng kernel Gaussian xung quanh tọa độ
        for i in range(kernel_size):
            for j in range(kernel_size):
                px = x + j - offset
                py = y + i - offset
                if 0 <= px < width_px and 0 <= py < height_px:
                    heatmap[py, px] += gaussian_kernel[i, j] * 50  # Tăng cường độ

# Làm mượt heatmap
heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)

# Normalize heatmap
heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
heatmap = heatmap.astype(np.uint8)

# Áp màu heatmap
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Overlay lên sân
court_img = cv2.imread('badminton_court_resized.jpg')
blended = cv2.addWeighted(court_img, 0.7, heatmap_color, 0.5, 0)

# Lưu kết quả
cv2.imwrite('court_with_heatmap.jpg', blended)
print("✅ Đã lưu ảnh heatmap vào court_with_heatmap.jpg")