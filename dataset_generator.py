import numpy as np
import cv2
import random
import os

class DatasetGenerator:
    def __init__(self, img_size=64):
        self.img_size = img_size
        # 6 shapes as requested
        self.shapes = ['line', 'circle', 'rectangle', 'triangle', 'ellipse', 'polygon']

    def add_hand_drawn_noise(self, img):
        # Halka sa Gaussian blur add kar rahe hain taake lines thori smooth/imperfect lagain
        # aur "hand-drawn" appearance de, CPU processing ko fast rakhne ke liye ye kaafi hai.
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        return blur

    def random_point(self):
        # Image ke borders se thora andar taake shapes bahar na cut hon
        margin = 5
        return (random.randint(margin, self.img_size - margin), 
                random.randint(margin, self.img_size - margin))

    def generate_single_shape(self):
        # Black background grayscale image
        img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        shape_type = random.choice(self.shapes)
        command = []

        thickness = random.randint(1, 2)
        color = 255 # White shape

        if shape_type == 'line':
            pt1, pt2 = self.random_point(), self.random_point()
            cv2.line(img, pt1, pt2, color, thickness)
            command = ['line', pt1[0], pt1[1], pt2[0], pt2[1]]

        elif shape_type == 'circle':
            center = self.random_point()
            # Radius limits taake boundaries se bahar na jaaye
            max_r = min(center[0], center[1], self.img_size - center[0], self.img_size - center[1])
            radius = random.randint(5, max(6, max_r))
            cv2.circle(img, center, radius, color, thickness)
            command = ['circle', center[0], center[1], radius]

        elif shape_type == 'rectangle':
            pt1 = self.random_point()
            pt2 = self.random_point()
            cv2.rectangle(img, pt1, pt2, color, thickness)
            x_min, y_min = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
            w, h = abs(pt2[0] - pt1[0]), abs(pt2[1] - pt1[1])
            command = ['rectangle', x_min, y_min, w, h]

        elif shape_type == 'triangle':
            pts = np.array([self.random_point(), self.random_point(), self.random_point()], np.int32)
            cv2.polylines(img, [pts.reshape((-1, 1, 2))], isClosed=True, color=color, thickness=thickness)
            command = ['triangle', pts[0][0], pts[0][1], pts[1][0], pts[1][1], pts[2][0], pts[2][1]]

        elif shape_type == 'ellipse':
            center = self.random_point()
            axes = (random.randint(8, 20), random.randint(4, 15))
            angle = random.randint(0, 180)
            cv2.ellipse(img, center, axes, angle, 0, 360, color, thickness)
            command = ['ellipse', center[0], center[1], axes[0], axes[1], angle]

        elif shape_type == 'polygon':
            num_points = random.randint(4, 6)
            pts = np.array([self.random_point() for _ in range(num_points)], np.int32)
            # Convex hull use kar rahe hain taake ajeeb cross hone wali polygon na banay
            pts = cv2.convexHull(pts)
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
            command = ['polygon'] + pts.flatten().tolist()

        img = self.add_hand_drawn_noise(img)
        return img, command

    def generate_dataset(self, num_samples, save_dir=None):
        images = []
        commands = []
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for i in range(num_samples):
            img, cmd = self.generate_single_shape()
            images.append(img)
            commands.append(cmd)
            
            # Agar testing ke liye images disk pe save karni hon
            if save_dir:
                cv2.imwrite(os.path.join(save_dir, f"{i}.png"), img)
                with open(os.path.join(save_dir, f"{i}.txt"), 'w') as f:
                    f.write(" ".join(map(str, cmd)))
                    
        return np.array(images), commands

if __name__ == "__main__":
    print("Generating dataset samples...")
    generator = DatasetGenerator(img_size=64)
    # 10 test samples generate karte hain verify karne ke liye
    imgs, cmds = generator.generate_dataset(10, save_dir="test_dataset_output")
    print("Dataset generation done! Check 'test_dataset_output' folder.")
    for i, c in enumerate(cmds):
        print(f"Sample {i} Labels: {c}")
