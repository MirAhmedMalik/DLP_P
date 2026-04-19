import numpy as np
import cv2
import torch


class InferenceEngine:
    """Production-grade Deterministic Geometric Vision Engine."""

    def __init__(self, model_path=None, img_size=64, output_size=256):
        self.img_size = img_size
        self.output_size = output_size
        self.scale = output_size / img_size
        print("Shape-to-Code Engine Ready.")

    def beam_search(self, image_tensor, max_len=40):
        img_np = (image_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
        return img_np, 1.0

    def _detect_lines(self, thresh_img):
        lines_found = []
        lines = cv2.HoughLinesP(thresh_img, 1, np.pi / 180,
                                threshold=15, minLineLength=10, maxLineGap=5)
        if lines is not None:
            for ln in lines:
                x1, y1, x2, y2 = ln[0]
                lines_found.append(['line', int(x1), int(y1), int(x2), int(y2)])
        return lines_found

    def post_process(self, raw_img):
        _, thresh = cv2.threshold(raw_img, 40, 255, cv2.THRESH_BINARY)
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        commands = []
        used_bboxes = []

        def clamp(v, lo=0, hi=self.img_size - 1):
            return max(lo, min(hi, int(v)))

        def is_dup(bx, by, bw, bh):
            for (ox, oy, ow, oh) in used_bboxes:
                if abs(bx-ox) < 3 and abs(by-oy) < 3 and abs(bw-ow) < 5 and abs(bh-oh) < 5:
                    return True
            return False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 12:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            if is_dup(bx, by, bw, bh):
                continue

            peri = cv2.arcLength(cnt, True)
            circularity = (4 * np.pi * area / (peri * peri)) if peri > 0 else 0
            aspect = bw / max(bh, 1)

            epsilon = 0.02 * peri
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            vert = len(approx)

            # convex_ratio: how much larger is the convex hull vs the raw contour area.
            # A circle drawn as a stroke has a huge void interior → hull_area >> contour_area → ratio HIGH
            # A polygon drawn as a stroke is mostly filled in the hull → ratio LOW
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            convex_ratio = hull_area / max(area, 1)

            def clamp_v(v): return clamp(v)

            # ─────────────────────────────────────────────────────────
            # CLASSIFICATION (vertex count is the primary authority)
            # ─────────────────────────────────────────────────────────

            if vert == 3:
                # Triangle
                pts = [clamp_v(v) for v in approx.flatten().tolist()[:6]]
                while len(pts) < 6:
                    pts.append(0)
                commands.append(['triangle'] + pts)
                used_bboxes.append((bx, by, bw, bh))

            elif vert == 4 and bw > 4 and bh > 4:
                # Rectangle
                commands.append(['rectangle', clamp_v(bx), clamp_v(by),
                                 clamp(bw, 1, 63), clamp(bh, 1, 63)])
                used_bboxes.append((bx, by, bw, bh))

            elif vert == 2 and circularity < 0.20:
                # Diagonal lines after dilation appear as vert=2, very low circularity
                commands.append(['line', clamp_v(bx), clamp_v(by),
                                 clamp_v(bx + bw), clamp_v(by + bh)])
                used_bboxes.append((bx, by, bw, bh))

            elif vert >= 5:
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)

                # Ellipse: smooth + clearly elongated bounding box
                if circularity > 0.50 and (aspect > 1.35 or aspect < 0.74) and len(cnt) >= 5:
                    try:
                        ell = cv2.fitEllipse(cnt)
                        (ecx, ecy), (MA, ma), angle = ell
                        commands.append(['ellipse', clamp_v(ecx), clamp_v(ecy),
                                         max(1, int(MA / 2)), max(1, int(ma / 2)), int(angle)])
                        used_bboxes.append((bx, by, bw, bh))
                    except Exception:
                        pass

                # Circle detection:
                # A true circle stroke at 64x64 scale always produces vert >= 8 after 0.02-epsilon
                # approximation (many tiny curved steps).
                # A hand-drawn polygon has fewer, larger straight-edge vertices (typically 5-7).
                # Also require high circularity > 0.82 and square bbox.
                elif vert >= 8 and circularity > 0.82 and 0.70 < aspect < 1.40:
                    if radius > 2:
                        commands.append(['circle', clamp_v(cx), clamp_v(cy),
                                         clamp(radius, 1, 30)])
                        used_bboxes.append((bx, by, bw, bh))

                # Polygon: 5-7 vertices, OR high-vertex blob that is not round enough for circle
                else:
                    pts = [clamp_v(v) for v in approx.flatten().tolist()]
                    if len(pts) >= 6:
                        commands.append(['polygon'] + pts)
                        used_bboxes.append((bx, by, bw, bh))

            else:
                # Remaining tiny blobs — treat as lines
                commands.append(['line', clamp_v(bx), clamp_v(by),
                                 clamp_v(bx + bw), clamp_v(by + bh)])
                used_bboxes.append((bx, by, bw, bh))

        if not commands:
            commands = self._detect_lines(thresh)

        commands.sort(key=lambda c: (c[2] if len(c) > 2 else 0, c[1] if len(c) > 1 else 0))
        return commands

    def reconstruct_image(self, commands):
        s = self.scale
        size = self.output_size
        img_out = np.zeros((size, size), dtype=np.uint8)
        c = 255
        t = 2

        def sc(v):
            return max(0, min(size - 1, int(v * s)))

        for cmd in commands:
            shape = cmd[0]
            vals = cmd[1:]
            try:
                if shape == 'line' and len(vals) >= 4:
                    cv2.line(img_out, (sc(vals[0]), sc(vals[1])),
                             (sc(vals[2]), sc(vals[3])), c, t)
                elif shape == 'circle' and len(vals) >= 3:
                    cv2.circle(img_out, (sc(vals[0]), sc(vals[1])),
                               max(1, int(vals[2] * s)), c, t)
                elif shape == 'rectangle' and len(vals) >= 4:
                    cv2.rectangle(img_out,
                                  (sc(vals[0]), sc(vals[1])),
                                  (sc(vals[0] + vals[2]), sc(vals[1] + vals[3])), c, t)
                elif shape == 'triangle' and len(vals) >= 6:
                    pts = np.array([[sc(vals[0]), sc(vals[1])],
                                    [sc(vals[2]), sc(vals[3])],
                                    [sc(vals[4]), sc(vals[5])]], np.int32)
                    cv2.polylines(img_out, [pts], True, c, t)
                elif shape == 'ellipse' and len(vals) >= 5:
                    axes = (max(1, int(vals[2] * s)), max(1, int(vals[3] * s)))
                    cv2.ellipse(img_out, (sc(vals[0]), sc(vals[1])),
                                axes, vals[4], 0, 360, c, t)
                elif shape == 'polygon' and len(vals) >= 6:
                    n = (len(vals) // 2) * 2
                    pts = np.array([[sc(vals[i]), sc(vals[i + 1])]
                                    for i in range(0, n, 2)], np.int32)
                    cv2.polylines(img_out, [pts], True, c, t)
            except Exception:
                continue

        return img_out
