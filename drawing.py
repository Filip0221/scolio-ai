from PIL import Image, ImageTk
import numpy as np
import cv2

# Stałe rysowania
POINT_RADIUS = 3
PAIR_Y_TOL = 30
LINE_EXTEND = 100

# Wyliczanie przesunięcia obrazu na canvasie
def center_offset(canvas, new_w, new_h, offset_x, offset_y):
    cw, ch = canvas.winfo_width(), canvas.winfo_height()
    return (cw - new_w)//2 + offset_x, (ch - new_h)//2 + offset_y


# funkcja odświeżania widoku
def redraw(canvas, original_img, zoom, offset_x, offset_y, pedicle_points,
           selected_lines, point_items, lines_coords):
    cw, ch = canvas.winfo_width(), canvas.winfo_height()
    if cw < 2 or ch < 2:
        return None, None

    def lines_match(p1, p2, q1, q2, tol=1e-6):
        a1, a2 = np.array(p1, dtype=float), np.array(p2, dtype=float)
        b1, b2 = np.array(q1, dtype=float), np.array(q2, dtype=float)
        return (
            (np.allclose(a1, b1, atol=tol) and np.allclose(a2, b2, atol=tol)) or
            (np.allclose(a1, b2, atol=tol) and np.allclose(a2, b1, atol=tol))
        )

    new_w = max(1, int(original_img.width * zoom))
    new_h = max(1, int(original_img.height * zoom))
    resized = original_img.resize((new_w, new_h), Image.LANCZOS)
    tk_img = ImageTk.PhotoImage(resized)

    canvas.delete("all")
    ox, oy = center_offset(canvas, new_w, new_h, offset_x, offset_y)
    bg_img_id = canvas.create_image(ox, oy, anchor="nw", image=tk_img)

    point_items.clear()
    lines_coords.clear()

    for i, (x, y, color, ptype) in enumerate(pedicle_points):
        sx, sy = x*zoom + ox, y*zoom + oy; r = POINT_RADIUS
        pid = canvas.create_oval(sx-r, sy-r, sx+r, sy+r, fill=color, outline="")
        point_items[pid] = i

    for ptype in ["top", "bottom"]:
        pts = [(x, y) for (x, y, _, t) in pedicle_points if t == ptype]
        pts.sort(key=lambda p: p[1])
        i = 0
        while i < len(pts) - 1:
            p1, p2 = pts[i], pts[i+1]
            if abs(p1[1] - p2[1]) < PAIR_Y_TOL:
                sx1, sy1 = p1[0]*zoom + ox, p1[1]*zoom + oy
                sx2, sy2 = p2[0]*zoom + ox, p2[1]*zoom + oy
                dx, dy = sx2 - sx1, sy2 - sy1
                length = np.hypot(dx, dy)
                if length > 0:
                    ex, ey = dx/length * LINE_EXTEND, dy/length * LINE_EXTEND
                    sx1 -= ex; sy1 -= ey; sx2 += ex; sy2 += ey
                cid = canvas.create_line(sx1, sy1, sx2, sy2, fill="yellow", width=2, dash=(4, 2))
                lines_coords.append(((p1[0], p1[1]), (p2[0], p2[1]), cid))
                i += 2
            else:
                i += 1
    matched_selected = []
    for p1, p2, cid in lines_coords:
        for sel_p1, sel_p2, _ in selected_lines:
            if lines_match(sel_p1, sel_p2, p1, p2):
                canvas.itemconfig(cid, fill="red", width=3, dash=())
                matched_selected.append((sel_p1, sel_p2, cid))
                break
    selected_lines[:] = matched_selected

    return tk_img, bg_img_id
