import os
import cv2
import numpy as np

os.makedirs('just_test', exist_ok=True)
img_size = 64
color = 255
thickness = 1

def blank():
    return np.zeros((img_size, img_size), dtype=np.uint8)

# 1. Line
img = blank()
cv2.line(img, (10, 10), (50, 50), color, thickness)
cv2.imwrite('just_test/line.png', cv2.GaussianBlur(img, (3,3), 0))

# 2. Circle
img = blank()
cv2.circle(img, (32, 32), 15, color, thickness)
cv2.imwrite('just_test/circle.png', cv2.GaussianBlur(img, (3,3), 0))

# 3. Rectangle
img = blank()
cv2.rectangle(img, (15, 15), (45, 45), color, thickness)
cv2.imwrite('just_test/rectangle.png', cv2.GaussianBlur(img, (3,3), 0))

# 4. Triangle
img = blank()
pts = np.array([[32, 10], [10, 50], [54, 50]], np.int32)
cv2.polylines(img, [pts], True, color, thickness)
cv2.imwrite('just_test/triangle.png', cv2.GaussianBlur(img, (3,3), 0))

# 5. Ellipse
img = blank()
cv2.ellipse(img, (32, 32), (20, 10), 45, 0, 360, color, thickness)
cv2.imwrite('just_test/ellipse.png', cv2.GaussianBlur(img, (3,3), 0))

# 6. Polygon
img = blank()
pts = np.array([[20, 10], [40, 10], [50, 30], [30, 50], [10, 30]], np.int32)
cv2.polylines(img, [pts], True, color, thickness)
cv2.imwrite('just_test/polygon.png', cv2.GaussianBlur(img, (3,3), 0))

# 7. Grid of Circles (Neuro-Symbolic Logic Test)
img = blank()
for i in range(3):
    for j in range(3):
        x = 15 + j * 15
        y = 15 + i * 15
        cv2.circle(img, (x, y), 4, color, thickness)
cv2.imwrite('just_test/grid_of_circles.png', cv2.GaussianBlur(img, (3,3), 0))

# 8. Multiple Miscellaneous Shapes in ONE Picture
img = blank()
cv2.circle(img, (20, 20), 8, color, thickness)
cv2.rectangle(img, (40, 10), (55, 25), color, thickness)
cv2.line(img, (10, 40), (25, 55), color, thickness)
cv2.ellipse(img, (45, 45), (10, 5), 30, 0, 360, color, thickness)
cv2.circle(img, (32, 32), 4, color, thickness)
cv2.imwrite('just_test/multiple_shapes.png', cv2.GaussianBlur(img, (3,3), 0))

print("Testing Images explicitly generated in: just_test folder!")
