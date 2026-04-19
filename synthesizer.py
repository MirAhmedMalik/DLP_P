import numpy as np

class CodeSynthesizer:
    def __init__(self, tolerance=6):
        self.tolerance = tolerance

    def almost_equal(self, a, b):
        return abs(a - b) <= self.tolerance

    def detect_progression(self, values):
        if len(values) < 2:
            return None
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        avg_diff = sum(diffs) / len(diffs)
        for d in diffs:
            if not self.almost_equal(d, avg_diff):
                return None
        return int(avg_diff)

    def group_shapes(self, commands):
        groups = {}
        for cmd in commands:
            name = cmd[0]
            if name not in groups:
                groups[name] = []
            groups[name].append(cmd[1:])
        return groups

    def synthesize_grid(self, name, items):
        if len(items) < 4:
            return None
        sorted_y = sorted(items, key=lambda i: i[1])

        rows = []
        current_row = [sorted_y[0]]
        for it in sorted_y[1:]:
            if self.almost_equal(it[1], current_row[-1][1]):
                current_row.append(it)
            else:
                rows.append(current_row)
                current_row = [it]
        rows.append(current_row)

        M = len(rows[0])
        if M < 2:
            return None
        for r in rows:
            if len(r) != M:
                return None

        N = len(rows)
        if N < 2:
            return None

        ys = [np.mean([it[1] for it in r]) for r in rows]
        dy = self.detect_progression(ys)
        if dy is None:
            return None

        r0_sorted = sorted(rows[0], key=lambda i: i[0])
        xs = [it[0] for it in r0_sorted]
        dx = self.detect_progression(xs)
        if dx is None:
            return None

        start_y = int(ys[0])
        start_x = int(xs[0])

        code = [
            f"    # Grid Pattern: {N} rows x {M} cols of {name}s",
            f"    for row in range({N}):",
            f"        for col in range({M}):",
            f"            y = {start_y} + row * {dy}",
            f"            x = {start_x} + col * {dx}",
        ]
        if name == 'circle':
            r_avg = int(np.mean([it[2] for it in items]))
            code.append(f"            cv2.circle(img, (x, y), {r_avg}, color, thickness)")
        elif name == 'rectangle':
            w_avg = int(np.mean([it[2] for it in items]))
            h_avg = int(np.mean([it[3] for it in items]))
            code.append(f"            cv2.rectangle(img, (x, y), (x+{w_avg}, y+{h_avg}), color, thickness)")
        else:
            return None
        return code

    def synthesize_1d_loop(self, name, items):
        if len(items) < 3:
            return None
        s_x = sorted(items, key=lambda i: i[0])
        xs = [it[0] for it in s_x]
        dx = self.detect_progression(xs)
        ys = [it[1] for it in s_x]
        y_avg = int(np.mean(ys))
        y_stable = all(self.almost_equal(y, y_avg) for y in ys)

        if dx is not None and abs(dx) > 0 and y_stable:
            start_x = int(xs[0])
            code = [
                f"    # 1D Array: {len(items)} {name}s in a row",
                f"    for i in range({len(items)}):",
                f"        x = {start_x} + i * {dx}",
            ]
            if name == 'circle':
                r_avg = int(np.mean([it[2] for it in items]))
                code.append(f"        cv2.circle(img, (x, {y_avg}), {r_avg}, color, thickness)")
            elif name == 'rectangle':
                w_avg = int(np.mean([it[2] for it in items]))
                h_avg = int(np.mean([it[3] for it in items]))
                code.append(f"        cv2.rectangle(img, (x, {y_avg}), (x+{w_avg}, {y_avg}+{h_avg}), color, thickness)")
            return code
        return None

    def generate_python_code(self, commands):
        if not commands:
            return "# No recognizable shapes detected."

        groups = self.group_shapes(commands)
        lines = [
            "import cv2",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "",
            "def draw_shapes():",
            "    # Coordinates are in 64x64 space (as detected)",
            "    img = np.zeros((64, 64), dtype=np.uint8)",
            "    color = 255",
            "    thickness = 1",
            "",
        ]

        for name, items in groups.items():
            grid_code = self.synthesize_grid(name, items)
            if grid_code:
                lines.extend(grid_code)
                lines.append("")
                continue

            loop_code = self.synthesize_1d_loop(name, items)
            if loop_code:
                lines.extend(loop_code)
                lines.append("")
                continue

            lines.append(f"    # Discrete {name} shapes")
            for vals in items:
                v = [str(int(x)) for x in vals]
                if name == 'circle' and len(v) >= 3:
                    lines.append(f"    cv2.circle(img, ({v[0]}, {v[1]}), {v[2]}, color, thickness)")
                elif name == 'line' and len(v) >= 4:
                    lines.append(f"    cv2.line(img, ({v[0]}, {v[1]}), ({v[2]}, {v[3]}), color, thickness)")
                elif name == 'rectangle' and len(v) >= 4:
                    lines.append(f"    cv2.rectangle(img, ({v[0]}, {v[1]}), ({v[0]}+{v[2]}, {v[1]}+{v[3]}), color, thickness)")
                elif name == 'triangle' and len(v) >= 6:
                    lines.append(f"    cv2.polylines(img, [np.array([[{v[0]},{v[1]}],[{v[2]},{v[3]}],[{v[4]},{v[5]}]])], True, color, thickness)")
                elif name == 'ellipse' and len(v) >= 5:
                    lines.append(f"    cv2.ellipse(img, ({v[0]},{v[1]}), ({v[2]},{v[3]}), {v[4]}, 0, 360, color, thickness)")
                elif name == 'polygon' and len(v) >= 6:
                    xys = ", ".join([f"[{v[i]},{v[i+1]}]" for i in range(0, len(v)-1, 2)])
                    lines.append(f"    cv2.polylines(img, [np.array([{xys}])], True, color, thickness)")
            lines.append("")

        lines.extend([
            "    # Upscale for sharp display after drawing at correct coords",
            "    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)",
            "    return img",
            "",
            "# Run and display:",
            "out = draw_shapes()",
            "plt.figure(figsize=(5,5))",
            "plt.imshow(out, cmap='gray')",
            "plt.axis('off')",
            "plt.title('Reconstructed Shape')",
            "plt.tight_layout()",
            "plt.show()",
        ])
        return "\n".join(lines)
