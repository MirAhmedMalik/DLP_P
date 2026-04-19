import numpy as np
import cv2
import random
import torch
from torch.utils.data import Dataset

VOCAB = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, 'line': 3, 'circle': 4, 'rectangle': 5, 
         'triangle': 6, 'ellipse': 7, 'polygon': 8}

for i in range(0, 361): 
    VOCAB[str(i)] = len(VOCAB)

IDX2VOCAB = {v: k for k, v in VOCAB.items()}

class GraphicsDataset(Dataset):
    # EXTENDED TO HANDLE 10 SHAPES FOR 3x3 GRIDS PREDICTION
    def __init__(self, num_samples, img_size=64, max_shapes=10):
        self.num_samples = num_samples
        self.img_size = img_size
        self.max_shapes = max_shapes
        self.shapes = ['line', 'circle', 'rectangle', 'triangle', 'ellipse', 'polygon']
        self.max_seq_len = max_shapes * 9 + 2 

    def random_point(self):
        m = 5
        return (random.randint(m, self.img_size - m), random.randint(m, self.img_size - m))

    def augment(self, img):
        img = cv2.GaussianBlur(img, (3, 3), 0)
        return img

    def generate_image_and_command(self):
        img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        num_shapes = random.randint(1, self.max_shapes)
        commands = ['<SOS>']
        
        t = random.randint(1, 2)
        c = random.randint(200, 255)
        
        # ADVANCED NEURO-SYMBOLIC DATA GENERATION
        # Model strictly needs to see structured grids to understand symmetry over random placements
        chance = random.random()
        
        if chance > 0.6: 
            # Inject Grids 40% of the time!
            grid_type = random.choice(['circle', 'rectangle'])
            N, M = random.randint(2, 3), random.randint(2, 3) 
            dx, dy = random.randint(10, 16), random.randint(10, 16)
            start_x = random.randint(5, max(6, self.img_size - M*dx - 5))
            start_y = random.randint(5, max(6, self.img_size - N*dy - 5))
            r = random.randint(3, 5)
            w, h = random.randint(5, 8), random.randint(5, 8)
            
            for i in range(N):
                for j in range(M):
                    y = start_y + i*dy
                    x = start_x + j*dx
                    if grid_type == 'circle':
                        cv2.circle(img, (x, y), r, c, t)
                        commands.extend(['circle', str(x), str(y), str(r)])
                    elif grid_type == 'rectangle':
                        cv2.rectangle(img, (x, y), (x+w, y+h), c, t)
                        commands.extend(['rectangle', str(x), str(y), str(w), str(h)])
                        
        else:
            # Fallback normal drawing logic for irregular datasets
            for _ in range(num_shapes):
                shape_type = random.choice(self.shapes)
                if shape_type == 'line':
                    p1, p2 = self.random_point(), self.random_point()
                    cv2.line(img, p1, p2, c, t)
                    commands.extend(['line', str(p1[0]), str(p1[1]), str(p2[0]), str(p2[1])])
                elif shape_type == 'circle':
                    center = self.random_point()
                    r = random.randint(5, max(6, min(center[0], center[1], self.img_size-center[0], self.img_size-center[1])))
                    cv2.circle(img, center, r, c, t)
                    commands.extend(['circle', str(center[0]), str(center[1]), str(r)])
                elif shape_type == 'rectangle':
                    p1, p2 = self.random_point(), self.random_point()
                    cv2.rectangle(img, p1, p2, c, t)
                    commands.extend(['rectangle', str(min(p1[0],p2[0])), str(min(p1[1],p2[1])), str(abs(p2[0]-p1[0])), str(abs(p2[1]-p1[1]))])
                elif shape_type == 'triangle':
                    pts = [self.random_point(), self.random_point(), self.random_point()]
                    cv2.polylines(img, [np.array(pts, np.int32)], True, c, t)
                    commands.extend(['triangle', str(pts[0][0]), str(pts[0][1]), str(pts[1][0]), str(pts[1][1]), str(pts[2][0]), str(pts[2][1])])
                elif shape_type == 'ellipse':
                    cen = self.random_point()
                    axes = (random.randint(8, 20), random.randint(4, 15))
                    ang = random.randint(0, 180)
                    cv2.ellipse(img, cen, axes, ang, 0, 360, c, t)
                    commands.extend(['ellipse', str(cen[0]), str(cen[1]), str(axes[0]), str(axes[1]), str(ang)])
                elif shape_type == 'polygon':
                    pts = cv2.convexHull(np.array([self.random_point() for _ in range(4)], np.int32))
                    cv2.polylines(img, [pts], True, c, t)
                    commands.extend(['polygon'] + [str(x) for x in pts.flatten()])

        commands.append('<EOS>')
        img = self.augment(img)
        token_indices = [VOCAB[tok] for tok in commands]
        
        if len(token_indices) < self.max_seq_len:
            token_indices.extend([VOCAB['<PAD>']] * (self.max_seq_len - len(token_indices)))
        else:
            token_indices = token_indices[:self.max_seq_len]
            
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return torch.tensor(img), torch.tensor(token_indices)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.generate_image_and_command()
