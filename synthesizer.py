import numpy as np

class CodeSynthesizer:
    def __init__(self, tolerance=6):
        # Allows for prediction pixel shifts / CNN Pooling noise
        self.tolerance = tolerance
        
    def almost_equal(self, a, b):
        return abs(a - b) <= self.tolerance
        
    def detect_progression(self, values):
        """Recognizes mathematical progression rules for For-Loops"""
        if len(values) < 2: return None
        diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
        avg_diff = sum(diffs)/len(diffs)
        for d in diffs:
            if not self.almost_equal(d, avg_diff):
                return None
        return int(avg_diff)

    def group_shapes(self, commands):
        shape_groups = {}
        for cmd in commands:
            name = cmd[0]
            if name not in shape_groups:
                shape_groups[name] = []
            shape_groups[name].append(cmd[1:])
        return shape_groups

    def synthesize_grid(self, name, items):
        """Engine to detect '3x3 grids' instead of '9 distinct items'"""
        if len(items) < 4: return None 
        sorted_y = sorted(items, key=lambda i: i[1])
        
        # Group by Rows
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
        if M < 2: return None
        for r in rows:
            if len(r) != M: return None
        
        N = len(rows)
        if N < 2: return None
        
        # Check delta Y
        ys = [np.mean([it[1] for it in r]) for r in rows]
        dy = self.detect_progression(ys)
        if dy is None: return None
        
        # Check delta X
        r0_sorted = sorted(rows[0], key=lambda i: i[0])
        xs = [it[0] for it in r0_sorted]
        dx = self.detect_progression(xs)
        if dx is None: return None
        
        start_y = int(ys[0])
        start_x = int(xs[0])
        
        code = []
        code.append(f"        # ---------------------------------------------")
        code.append(f"        # ✨ DETECTED {N}x{M} SMART GRID PATTERN OF {name.upper()}S ✨")
        code.append(f"        # ---------------------------------------------")
        code.append(f"        for row in range({N}):")
        code.append(f"            for col in range({M}):")
        code.append(f"                y = {start_y} + row * {dy}")
        code.append(f"                x = {start_x} + col * {dx}")
        
        if name == 'circle':
            r_avg = int(np.mean([it[2] for it in items]))
            code.append(f"                cv2.circle(img, (x, y), {r_avg}, color, thickness)")
        elif name == 'rectangle':
            w_avg = int(np.mean([it[2] for it in items]))
            h_avg = int(np.mean([it[3] for it in items]))
            code.append(f"                cv2.rectangle(img, (x, y), (x+{w_avg}, y+{h_avg}), color, thickness)")
        else:
            return None 
            
        return code

    def synthesize_1d_loop(self, name, items):
        if len(items) < 3: return None
        s_x = sorted(items, key=lambda i: i[0])
        xs = [it[0] for it in s_x]
        dx = self.detect_progression(xs)
        
        ys = [it[1] for it in s_x]
        y_avg = int(np.mean(ys))
        y_stable = all(self.almost_equal(y, y_avg) for y in ys)
        
        if dx is not None and abs(dx) > 0 and y_stable:
            start_x = int(xs[0])
            code = []
            code.append(f"        # ✨ Detected 1D Horizontal Array Pattern ✨")
            code.append(f"        for i in range({len(items)}):")
            code.append(f"            x = {start_x} + i * {dx}")
            if name == 'circle':
                r_avg = int(np.mean([it[2] for it in items]))
                code.append(f"            cv2.circle(img, (x, {y_avg}), {r_avg}, color, thickness)")
            elif name == 'rectangle':
                w_avg = int(np.mean([it[2] for it in items]))
                h_avg = int(np.mean([it[3] for it in items]))
                code.append(f"            cv2.rectangle(img, (x, {y_avg}), (x+{w_avg}, {y_avg}+{h_avg}), color, thickness)")
            return code
        return None

    def generate_python_code(self, commands):
        if not commands:
            return "No recognizable shapes found to format."
            
        groups = self.group_shapes(commands)
        code_lines = []
        code_lines.append("# --- Automatically Synthesized Executable LaTeX/Program Logic ---")
        code_lines.append("import cv2")
        code_lines.append("import numpy as np")
        code_lines.append("\ndef draw_inferred_shapes():")
        code_lines.append("    img = np.zeros((64, 64), dtype=np.uint8)")
        code_lines.append("    color = 255; thickness = 1\n")
        
        for name, items in groups.items():
            grid_code = self.synthesize_grid(name, items)
            if grid_code:
                code_lines.extend(grid_code)
                continue
                
            line_code = self.synthesize_1d_loop(name, items)
            if line_code:
                code_lines.extend(line_code)
                continue
                
            # Basic fallback discrete rendering
            code_lines.append(f"    # Drawing Discrete {name} elements")
            for vals in items:
                v = [str(x) for x in vals]
                if name == 'circle':
                    code_lines.append(f"    cv2.circle(img, ({v[0]}, {v[1]}), {v[2]}, color, thickness)")
                elif name == 'line':
                    code_lines.append(f"    cv2.line(img, ({v[0]}, {v[1]}), ({v[2]}, {v[3]}), color, thickness)")
                elif name == 'rectangle':
                    code_lines.append(f"    cv2.rectangle(img, ({v[0]}, {v[1]}), ({v[0]}+{v[2]}, {v[1]}+{v[3]}), color, thickness)")
                elif name == 'triangle':
                    code_lines.append(f"    cv2.polylines(img, [np.array([[{v[0]},{v[1]}], [{v[2]},{v[3]}], [{v[4]},{v[5]}]])], True, color, thickness)")
                elif name == 'ellipse':
                    code_lines.append(f"    cv2.ellipse(img, ({v[0]},{v[1]}), ({v[2]},{v[3]}), {v[4]}, 0, 360, color, thickness)")
                elif name == 'polygon':
                    xys = ",".join([f"[{v[i]},{v[i+1]}]" for i in range(0, len(v)-1, 2)])
                    code_lines.append(f"    cv2.polylines(img, [np.array([{xys}])], True, color, thickness)")
        
        # Function Return statement & Jupyter plotting
        code_lines.append("\n    return img")
        code_lines.append("\n# Jupyter / Python execution code to show drawing:")
        code_lines.append("import matplotlib.pyplot as plt")
        code_lines.append("out_img = draw_inferred_shapes()")
        code_lines.append("plt.imshow(out_img, cmap='gray')")
        code_lines.append("plt.axis('off')")
        code_lines.append("plt.show()")
        
        return "\n".join(code_lines)
