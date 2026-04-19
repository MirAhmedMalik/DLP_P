import os
import cv2
import numpy as np

os.makedirs('just_test', exist_ok=True)
s = 64
c = 255
t = 1

def blank():
    return np.zeros((s, s), dtype=np.uint8)

def save(name, img):
    # No artificial blur - keep crisp
    cv2.imwrite(f'just_test/{name}.png', img)

# --- Individual shapes ---
img = blank(); cv2.circle(img, (32,32), 18, c, t); save('circle', img)
img = blank(); cv2.rectangle(img, (10,10), (52,45), c, t); save('rectangle', img)
img = blank(); cv2.line(img, (8,8), (56,56), c, t); save('line', img)
img = blank(); cv2.line(img, (8,50), (56,15), c, t); save('line_diagonal2', img)
img = blank()
cv2.polylines(img, [np.array([[32,6],[6,58],[58,58]])], True, c, t)
save('triangle', img)
img = blank(); cv2.ellipse(img, (32,32), (24,10), 0, 0, 360, c, t); save('ellipse_wide', img)
img = blank(); cv2.ellipse(img, (32,32), (10,22), 45, 0, 360, c, t); save('ellipse_tilted', img)
img = blank()
pts = np.array([[32,6],[52,18],[58,42],[42,58],[22,58],[6,42],[12,18]])
cv2.polylines(img, [pts], True, c, t)
save('polygon_7sides', img)
img = blank()
pts = np.array([[32,6],[55,26],[46,54],[18,54],[9,26]])
cv2.polylines(img, [pts], True, c, t)
save('polygon_pentagon', img)

# --- Multi-shape images ---
img = blank()
cv2.circle(img, (18,18), 10, c, t)
cv2.rectangle(img, (34,8), (58,30), c, t)
save('multi_circle_rectangle', img)

img = blank()
cv2.line(img, (5,32), (58,32), c, t)
cv2.line(img, (32,5), (32,58), c, t)
save('multi_cross_lines', img)

img = blank()
cv2.circle(img, (16,16), 8, c, t)
cv2.circle(img, (48,16), 8, c, t)
cv2.circle(img, (16,48), 8, c, t)
cv2.circle(img, (48,48), 8, c, t)
save('multi_four_circles', img)

img = blank()
cv2.circle(img, (20,20), 8, c, t)
cv2.rectangle(img, (36,12), (58,32), c, t)
cv2.line(img, (8,44), (30,58), c, t)
cv2.ellipse(img, (48,50), (10,5), 0, 0, 360, c, t)
save('multi_4shapes', img)

img = blank()
cv2.circle(img, (15,15), 7, c, t)
cv2.circle(img, (32,15), 7, c, t)
cv2.circle(img, (49,15), 7, c, t)
cv2.circle(img, (15,32), 7, c, t)
cv2.circle(img, (32,32), 7, c, t)
cv2.circle(img, (49,32), 7, c, t)
cv2.circle(img, (15,49), 7, c, t)
cv2.circle(img, (32,49), 7, c, t)
cv2.circle(img, (49,49), 7, c, t)
save('grid_3x3_circles', img)

img = blank()
for i in range(4):
    x = 10 + i*14
    cv2.circle(img, (x, 32), 5, c, t)
save('array_4circles_row', img)

img = blank()
cv2.rectangle(img, (5,5), (28,28), c, t)
cv2.rectangle(img, (34,5), (58,28), c, t)
cv2.rectangle(img, (5,34), (28,58), c, t)
cv2.rectangle(img, (34,34), (58,58), c, t)
save('grid_2x2_rectangles', img)

print('All test images created in just_test/')
