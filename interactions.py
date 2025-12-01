import numpy as np
from drawing import redraw, center_offset
from cobb_angle import cobb_angle

# styl lini w zaleznosci czy wybrana
def set_line_style(canvas, cid, selected):
    if selected:
        canvas.itemconfig(cid, fill="red", width=3, dash=())
    else:
        canvas.itemconfig(cid, fill="yellow", width=2, dash=(4, 2))

# czyści wybór lini pomocniczych
def reset_helper_items(canvas, helper_items):

    for hid in helper_items:
        canvas.delete(hid)
    helper_items.clear()

# Zwraca kwadrat odległości punktu od odcinka
def point_segment_distance_sq(px, py, x1, y1, x2, y2):
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    seg_len_sq = vx * vx + vy * vy
    if seg_len_sq <= 1e-9:
        return (px - x1) ** 2 + (py - y1) ** 2
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len_sq))
    proj_x = x1 + t * vx
    proj_y = y1 + t * vy
    return (px - proj_x) ** 2 + (py - proj_y) ** 2

#Sprawdza czy dwie linie (niezależnie od kolejności punktów) reprezentują to samo
def lines_match(p1, p2, q1, q2, tol=1e-6):
    a1, a2 = np.array(p1, dtype=float), np.array(p2, dtype=float)
    b1, b2 = np.array(q1, dtype=float), np.array(q2, dtype=float)
    return (
        (np.allclose(a1, b1, atol=tol) and np.allclose(a2, b2, atol=tol)) or
        (np.allclose(a1, b2, atol=tol) and np.allclose(a2, b1, atol=tol))
    )

# Znajdź linię która leży w zadanym progu od kliknięcia
def find_nearest_line(event, canvas, lines_coords, max_dist=12):
    best = None
    best_d2 = max_dist * max_dist
    for p1, p2, cid in lines_coords:
        coords = canvas.coords(cid)
        if not coords or len(coords) < 4:
            continue
        x1, y1, x2, y2 = coords[:4]
        d2 = point_segment_distance_sq(event.x, event.y, x1, y1, x2, y2)
        if d2 < best_d2:
            best_d2 = d2
            best = (p1, p2, cid)
    return best


#PODSTAWOWE TRYBY I AKCJE

#tryb dodawania punktów
def set_add_mode(color, ptype, result_label, add_mode_ref, remove_mode_ref, select_mode_ref, drag_state):
    add_mode_ref[0] = (color, ptype)
    remove_mode_ref[0] = None
    select_mode_ref[0] = False
    drag_state["allow_drag"] = False
    drag_state["drag_point_id"] = None
    drag_state["drag_point_idx"] = None
    result_label.config(
        text=f"Tryb dodawania: kliknij na obraz, aby dodać punkt {ptype} ({color})."
    )

#tryb usuwania punktów
def set_remove_mode(result_label, remove_mode_ref, add_mode_ref, select_mode_ref, drag_state):
    remove_mode_ref[0] = "any"
    add_mode_ref[0] = None
    select_mode_ref[0] = False
    drag_state["allow_drag"] = False
    drag_state["drag_point_id"] = None
    drag_state["drag_point_idx"] = None
    result_label.config(text="Tryb usuwania: kliknij punkt, aby go usunąć.")

#tryb wyboru linii
def start_select_mode(result_label, select_mode_ref, selected_lines_ref, add_mode_ref, remove_mode_ref, drag_state):
    select_mode_ref[0] = True
    selected_lines_ref.clear()
    add_mode_ref[0] = None
    remove_mode_ref[0] = None
    drag_state["allow_drag"] = False
    drag_state["drag_point_id"] = None
    drag_state["drag_point_idx"] = None
    result_label.config(text="Tryb wyboru: kliknij dwie żółte linie do Cobba.")

#tryb przesuwania punktów
def enable_drag_mode(result_label, drag_state, add_mode_ref, remove_mode_ref, select_mode_ref):
    add_mode_ref[0] = None
    remove_mode_ref[0] = None
    select_mode_ref[0] = False
    drag_state["allow_drag"] = True
    drag_state["drag_point_id"] = None
    drag_state["drag_point_idx"] = None
    result_label.config(text="Tryb przesuwania: przeciągnij wybrany punkt, aby go przesunąć.")

# obliczanie kata cobba i rysowanie lini
def compute_cobb(canvas, original_img, zoom, selected_lines, result_label, helper_items):
    reset_helper_items(canvas, helper_items)

    if len(selected_lines) != 2:
        return

    ang = cobb_angle(selected_lines[0], selected_lines[1])
    result_label.config(text=f"Kąt Cobba: {ang:.2f}°")

    (a1, a2, _) = selected_lines[0]
    (b1, b2, _) = selected_lines[1]

    mid1 = ((a1[0] + a2[0]) / 2.0, (a1[1] + a2[1]) / 2.0)
    mid2 = ((b1[0] + b2[0]) / 2.0, (b1[1] + b2[1]) / 2.0)

    d1 = np.array([a2[0] - a1[0], a2[1] - a1[1]], dtype=float)
    d2 = np.array([b2[0] - b1[0], b2[1] - b1[1]], dtype=float)
    n1 = np.array([-d1[1], d1[0]], dtype=float)
    n2 = np.array([-d2[1], d2[0]], dtype=float)
    if np.linalg.norm(n1) < 1e-9 or np.linalg.norm(n2) < 1e-9:
        return
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)

    new_w = int(original_img.width * zoom)
    new_h = int(original_img.height * zoom)
    ox, oy = center_offset(canvas, new_w, new_h, 0, 0)

    base_len = np.clip(np.linalg.norm(np.array(mid2) - np.array(mid1)) * 0.5, 60.0, 200.0)
    extend_len = np.clip(np.linalg.norm(np.array(mid2) - np.array(mid1)) * 0.25, 30.0, 120.0)

    cross = ((mid1[0] + mid2[0]) / 2.0, (mid1[1] + mid2[1]) / 2.0)

    def project_point(point, line_point, line_vec):
        lv = line_vec / np.linalg.norm(line_vec)
        return line_point + np.dot(np.array(point) - np.array(line_point), lv) * lv

    def build_perpendicular(mid_point, line_vec, normal_vec, cross_point):
        start = project_point(cross_point, mid_point, line_vec)
        dir_vec = np.array(cross_point) - np.array(start)
        if np.linalg.norm(dir_vec) < 1e-6:
            dir_vec = normal_vec
        dir_unit = dir_vec / np.linalg.norm(dir_vec)
        start_offset = start
        end = cross_point + dir_unit * extend_len
        return start_offset, end

    top_pair = (mid1, d1, n1)
    bottom_pair = (mid2, d2, n2)
    if mid2[1] < mid1[1]:
        top_pair, bottom_pair = bottom_pair, top_pair

    top_start, top_end = build_perpendicular(*top_pair, cross)
    bottom_start, bottom_end = build_perpendicular(*bottom_pair, np.array(cross) - 1e-9)

    if top_end[1] < top_start[1]:
        top_end = (top_start[0] + (top_end[0] - top_start[0]), top_start[1] + abs(top_end[1] - top_start[1]))

    if bottom_end[1] > bottom_start[1]:
        bottom_end = (bottom_start[0] + (bottom_end[0] - bottom_start[0]), bottom_start[1] - abs(bottom_end[1] - bottom_start[1]))

    line1_start, line1_end = top_start, top_end
    line2_start, line2_end = bottom_start, bottom_end

    sx1_start, sy1_start = line1_start[0] * zoom + ox, line1_start[1] * zoom + oy
    sx1_end, sy1_end = line1_end[0] * zoom + ox, line1_end[1] * zoom + oy
    sx2_start, sy2_start = line2_start[0] * zoom + ox, line2_start[1] * zoom + oy
    sx2_end, sy2_end = line2_end[0] * zoom + ox, line2_end[1] * zoom + oy

    faint_red = "#ff4d4d"
    cid1 = canvas.create_line(sx1_start, sy1_start, sx1_end, sy1_end, fill=faint_red, width=1, dash=())
    cid2 = canvas.create_line(sx2_start, sy2_start, sx2_end, sy2_end, fill=faint_red, width=1, dash=())

    helper_items.extend([cid1, cid2])

    tip_avg = (
        (line1_end[0] + line2_end[0]) / 2.0,
        (line1_end[1] + line2_end[1]) / 2.0
    )
    cx = tip_avg[0] * zoom + ox
    cy = tip_avg[1] * zoom + oy
    r = 5
    # mark = canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="red", width=2)
    label = canvas.create_text(
        cx + 20, cy - 20, text=f"{ang:.1f}°", fill="red", font=("Arial", 14, "bold")
    )
    helper_items.extend([label])

# wyczyszczenie wyboru i reset widoku
def clear_selection(canvas, selected_lines, helper_items, result_label,
                    select_mode_ref, zoom_ref, offset_x_ref, offset_y_ref,
                    pedicle_points, point_items, lines_coords, original_img,
                    add_mode_ref, remove_mode_ref, drag_state):
    for _, _, cid in selected_lines:
        set_line_style(canvas, cid, selected=False)
    reset_helper_items(canvas, helper_items)
    selected_lines.clear()
    select_mode_ref[0] = False
    add_mode_ref[0] = None
    remove_mode_ref[0] = None
    drag_state["allow_drag"] = False
    drag_state["drag_point_id"] = None
    drag_state["drag_point_idx"] = None
    zoom_ref[0] = 1.0
    offset_x_ref[0] = 0
    offset_y_ref[0] = 0
    tk_img, _ = redraw(
        canvas, original_img, zoom_ref[0], offset_x_ref[0], offset_y_ref[0],
        pedicle_points, selected_lines, point_items, lines_coords
    )
    if tk_img is not None:
        canvas.image_ref = tk_img
    result_label.config(text="Przywrócono widok początkowy.")


#ZOOM I COFNIĘCIE

# zastosowanie zoomu
def apply_zoom(factor, zoom_ref, canvas, original_img, offset_x_ref, offset_y_ref,
               pedicle_points, selected_lines, point_items, lines_coords):
    new_zoom = float(np.clip(zoom_ref[0] * factor, 0.2, 6.0))
    if abs(new_zoom - zoom_ref[0]) > 1e-6:
        zoom_ref[0] = new_zoom
        tk_img, _ = redraw(
            canvas, original_img, zoom_ref[0],
            offset_x_ref[0], offset_y_ref[0],
            pedicle_points, selected_lines,
            point_items, lines_coords
        )
        canvas.image_ref = tk_img  # zapobiega znikaniu obrazu

# OBSŁUGA MYSZY

# wspolrzedne obrazu z wspolrzednych canvasa
def canvas_to_image(canvas, xc, yc, original_img, zoom_ref, offset_x_ref, offset_y_ref):
    new_w = int(original_img.width * zoom_ref[0])
    new_h = int(original_img.height * zoom_ref[0])
    cw, ch = canvas.winfo_width(), canvas.winfo_height()
    ox = (cw - new_w)//2 + offset_x_ref[0]
    oy = (ch - new_h)//2 + offset_y_ref[0]
    return (xc - ox) / zoom_ref[0], (yc - oy) / zoom_ref[0]

# znajdowanie najbliższego punktu
def find_nearest_point(canvas, event, point_items, max_px=8):
    min_d2 = (max_px + 1)**2
    best = None
    for pid, idx in point_items.items():
        x1, y1, x2, y2 = canvas.coords(pid)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        d2 = (event.x - cx)**2 + (event.y - cy)**2
        if d2 < min_d2:
            min_d2 = d2
            best = (pid, idx)
    return best if best else (None, None)


# klik myszki dodanie przesuneici punktu
def on_mouse_down(event, canvas, pedicle_points, manual_points, point_items,
                  add_mode_ref, remove_mode_ref, drag_state, offset_x_ref, offset_y_ref, original_img,
                  result_label, select_mode_ref, selected_lines, lines_coords, helper_items):
    pid, idx = find_nearest_point(canvas, event, point_items)
    if add_mode_ref[0] is not None:
        x_img, y_img = canvas_to_image(canvas, event.x, event.y, original_img, drag_state["zoom"], offset_x_ref, offset_y_ref)
        new_point = (x_img, y_img, add_mode_ref[0][0], add_mode_ref[0][1])
        pedicle_points.append(new_point)
        manual_points.append(new_point)
        add_mode_ref[0] = None
        result_label.config(text="Dodano punkt.")
        canvas.event_generate("<<Redraw>>")
    elif remove_mode_ref[0] is not None:
        if pid is None:
            result_label.config(text="Kliknij dokładnie na punkt, aby usunąć.")
            return
        point = pedicle_points[idx]
        color = point[2]
        mode_color = remove_mode_ref[0]
        if mode_color != "any" and color != mode_color:
            result_label.config(text="Wybrany punkt ma inny kolor.")
            return
        removed = pedicle_points.pop(idx)
        try:
            manual_points.remove(removed)
        except ValueError:
            pass
        result_label.config(text=f"Usunięto punkt {color}.")
        canvas.event_generate("<<Redraw>>")
    elif select_mode_ref[0]:
        line_hit = find_nearest_line(event, canvas, lines_coords)
        if line_hit is None:
            result_label.config(text="Kliknij blisko żółtej linii, aby ją wybrać.")
            return

        p1, p2, cid = line_hit
        existing_idx = next(
            (i for i, (sel_p1, sel_p2, _) in enumerate(selected_lines)
             if lines_match(sel_p1, sel_p2, p1, p2)),
            None
        )
        if existing_idx is not None:
            set_line_style(canvas, selected_lines[existing_idx][2], selected=False)
            selected_lines.pop(existing_idx)
            reset_helper_items(canvas, helper_items)
            if selected_lines:
                result_label.config(text="Pozostała jedna linia. Wybierz jeszcze jedną.")
            else:
                result_label.config(text="Usunięto linię z wyboru.")
        else:
            if len(selected_lines) == 2:
                set_line_style(canvas, selected_lines[0][2], selected=False)
                selected_lines.pop(0)
            selected_lines.append((p1, p2, cid))
            set_line_style(canvas, cid, selected=True)
            if len(selected_lines) < 2:
                result_label.config(text="Wybrano linię. Kliknij drugą, aby obliczyć kąt.")
            else:
                result_label.config(text="Wybrano dwie linie. Liczę kąt Cobba.")
        if len(selected_lines) == 2:
            compute_cobb(canvas, original_img, drag_state["zoom"][0], selected_lines, result_label, helper_items)
        else:
            reset_helper_items(canvas, helper_items)
        return
    elif drag_state.get("allow_drag") and pid is not None:
        drag_state["drag_point_id"] = pid
        drag_state["drag_point_idx"] = idx
    elif pid is not None:
        result_label.config(text="Przesuwanie punktów jest wyłączone. Użyj przycisku trybu przesuwania.")
    else:
        drag_state["is_panning"] = True
        drag_state["pan_start"] = (event.x, event.y)
        drag_state["offset_start"] = (offset_x_ref[0], offset_y_ref[0])

# ruch myszy
def on_mouse_move(event, canvas, pedicle_points, drag_state, offset_x_ref, offset_y_ref):
    if drag_state["drag_point_id"] is not None and drag_state["drag_point_idx"] is not None:
        r = 3
        canvas.coords(drag_state["drag_point_id"], event.x - r, event.y - r, event.x + r, event.y + r)
    elif drag_state["is_panning"]:
        dx, dy = event.x - drag_state["pan_start"][0], event.y - drag_state["pan_start"][1]
        offset_x_ref[0] = drag_state["offset_start"][0] + dx
        offset_y_ref[0] = drag_state["offset_start"][1] + dy
        canvas.event_generate("<<Redraw>>")

# puszczenie przycisku myszy
def on_mouse_up(event, canvas, pedicle_points, drag_state, original_img, zoom_ref, offset_x_ref, offset_y_ref):
    if drag_state.get("drag_point_id") is not None and drag_state.get("drag_point_idx") is not None:
        try:
            x1, y1, x2, y2 = canvas.coords(drag_state["drag_point_id"])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            x_img, y_img = canvas_to_image(canvas, cx, cy, original_img, zoom_ref, offset_x_ref, offset_y_ref)

            idx = drag_state["drag_point_idx"]
            if 0 <= idx < len(pedicle_points):
                old = pedicle_points[idx]
                color = old[2] if len(old) > 2 else None
                ptype = old[3] if len(old) > 3 else None
                pedicle_points[idx] = (x_img, y_img, color, ptype)
        except Exception:
            pass
        finally:
            drag_state["drag_point_id"] = None
            drag_state["drag_point_idx"] = None
            try:
                canvas.event_generate("<<Redraw>>", when="tail")
            except Exception:
                pass

    drag_state["is_panning"] = False

