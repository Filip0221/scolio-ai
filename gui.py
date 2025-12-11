import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import os

from model_loader import load_model_and_points
from drawing import redraw
from interactions import (
    set_add_mode, set_remove_mode, start_select_mode, enable_drag_mode,
    compute_cobb, clear_selection,
    apply_zoom,
    on_mouse_down, on_mouse_move, on_mouse_up
)


def start_gui():
    # Ścieżka do modelu (możesz zmienić)
    MODEL_PATH = "runs/segment/pedicle_yolo11_retrain/weights/best.pt"

    # Stan aplikacji (zamykamy w liście/ dict by mieć mutowalny ref w lambda)
    current = {
        "image_path": None,
        "img_array": None,
        "pedicle_points": [],
        "original_img": None,
        "tk_img": None,
        "bg_img_id": None,
        "zoom": 1.0,
        "offset_x": 0,
        "offset_y": 0,
    }

    # Okno
    root = tk.Tk()
    root.title("Oblicz kąt cobba")
    root.geometry("900x700")

    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)
    canvas = tk.Canvas(main_frame, bg="black", cursor="tcross")
    canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    bottom_frame = ttk.Frame(main_frame, padding=6)
    bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

    label_frame = ttk.Frame(bottom_frame)
    label_frame.pack(side=tk.TOP, fill=tk.X)
    result_label = ttk.Label(label_frame, text="Lewy: dodaj/usuń punkt lub przesuwaj obraz. Scroll: zoom.")
    result_label.pack(side=tk.TOP, pady=2)

    buttons_frame = ttk.Frame(bottom_frame)
    buttons_frame.pack(side=tk.TOP, fill=tk.X, pady=4)

    # Stan przeciągania i przesuwania — musi być przed przyciskami, bo używane w callbackach
    drag_state = {
        "drag_point_id": None,
        "drag_point_idx": None,
        "is_panning": False,
        "pan_start": (0, 0),
        "offset_start": (0, 0),
        "zoom": [current["zoom"]],
        "allow_drag": False
    }

    # Inne refy
    zoom_ref = [1.0]
    offset_x_ref = [0]
    offset_y_ref = [0]
    selected_lines = []
    helper_items = []
    manual_points = []
    point_items = {}
    remove_mode_ref = [None]
    lines_coords = []
    add_mode_ref = [None]
    select_mode_ref = [False]

    # Przyciski - layout
    top_buttons = ttk.Frame(buttons_frame)
    top_buttons.pack(side=tk.TOP, fill=tk.X, pady=2)
    bottom_buttons = ttk.Frame(buttons_frame)
    bottom_buttons.pack(side=tk.TOP, fill=tk.X, pady=2)

    # Funkcja do wgrania obrazu i wyświetlenia
    def upload_image():
        path = filedialog.askopenfilename(
            title="Wybierz zdjęcie",
            filetypes=[("Images", "*.jpg *.jpeg *.png")]
        )
        if not path:
            return
        # Załaduj obraz
        img_arr = cv2.imread(path)
        if img_arr is None:
            result_label.config(text="Błąd: nie można wczytać obrazu.")
            return
        img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)

        # Zaktualizuj stan
        current["image_path"] = path
        current["img_array"] = img_arr
        current["pedicle_points"] = []  # wyczyść poprzednie punkty po nowym obrazie
        current["original_img"] = pil

        # Reset offset/zoom
        zoom_ref[0] = 1.0
        offset_x_ref[0] = 0
        offset_y_ref[0] = 0
        drag_state["zoom"][0] = 1.0

        # Rysuj
        tk_img, bg_id = redraw(
            canvas, pil, zoom_ref[0], offset_x_ref[0], offset_y_ref[0],
            current["pedicle_points"], selected_lines, point_items, lines_coords
        )
        current["tk_img"] = tk_img
        current["bg_img_id"] = bg_id
        canvas.image_ref = tk_img
        result_label.config(text=f"Wczytano: {os.path.basename(path)}")

    # Funkcja do uruchomienia modelu na aktualnym obrazie
    def run_model_on_current_image():
        if not current["image_path"]:
            result_label.config(text="Najpierw wgraj zdjęcie.")
            return

        result_label.config(text="Uruchamiam model...")
        root.update_idletasks()

        try:
            img_arr, ped_points = load_model_and_points(MODEL_PATH, current["image_path"])
        except Exception as e:
            result_label.config(text=f"Błąd modelu: {e}")
            return

        # Jeśli load_model zwrócił obraz, użyj go, w przeciwnym razie pozostaw oryginalny
        if isinstance(img_arr, (list, tuple)):
            pass
        current["img_array"] = img_arr
        current["pedicle_points"] = ped_points or []

        # Zaktualizuj original_img jeśli img_array istnieje
        if current["img_array"] is not None:
            img_rgb = cv2.cvtColor(current["img_array"], cv2.COLOR_BGR2RGB)
            current["original_img"] = Image.fromarray(img_rgb)

        # Rysuj ponownie z punktami
        tk_img, bg_id = redraw(
            canvas, current["original_img"], zoom_ref[0], offset_x_ref[0], offset_y_ref[0],
            current["pedicle_points"], selected_lines, point_items, lines_coords
        )
        current["tk_img"] = tk_img
        current["bg_img_id"] = bg_id
        canvas.image_ref = tk_img
        result_label.config(text=f"Gotowe — znaleziono {len(current['pedicle_points'])} punktów.")

    # Przyciski zoom/clear/quit/upload/run
    zoom_in_btn = ttk.Button(top_buttons, text="+", command=lambda: apply_zoom(
        1.2, zoom_ref, canvas, current["original_img"], offset_x_ref, offset_y_ref,
        current["pedicle_points"], selected_lines, point_items, lines_coords))
    zoom_out_btn = ttk.Button(top_buttons, text="-", command=lambda: apply_zoom(
        1/1.2, zoom_ref, canvas, current["original_img"], offset_x_ref, offset_y_ref,
        current["pedicle_points"], selected_lines, point_items, lines_coords))
    clear_btn = ttk.Button(top_buttons, text="Wyczyść wybór", command=lambda: clear_selection(
        canvas, selected_lines, helper_items, result_label,
        select_mode_ref, zoom_ref, offset_x_ref, offset_y_ref,
        current["pedicle_points"], point_items, lines_coords, current["original_img"],
        add_mode_ref, remove_mode_ref, drag_state))
    quit_button = ttk.Button(top_buttons, text="Quit", command=root.destroy)
    upload_btn = ttk.Button(top_buttons, text="Wgraj zdjęcie", command=upload_image)
    run_model_btn = ttk.Button(top_buttons, text="Wyznacz punkty", command=run_model_on_current_image)

    for b in [zoom_in_btn, zoom_out_btn, clear_btn, quit_button, upload_btn, run_model_btn]:
        b.pack(side=tk.LEFT, padx=5)

    # Dolne przyciski interakcji punktów
    add_green_btn = ttk.Button(bottom_buttons, text="Dodaj zielony",
        command=lambda: set_add_mode("green", "top", result_label, add_mode_ref, remove_mode_ref, select_mode_ref, drag_state))
    add_red_btn = ttk.Button(bottom_buttons, text="Dodaj czerwony",
        command=lambda: set_add_mode("red", "bottom", result_label, add_mode_ref, remove_mode_ref, select_mode_ref, drag_state))
    remove_btn = ttk.Button(bottom_buttons, text="Usuń punkt",
        command=lambda: set_remove_mode(result_label, remove_mode_ref, add_mode_ref, select_mode_ref, drag_state))
    move_btn = ttk.Button(bottom_buttons, text="Przesuwaj punkt",
        command=lambda: enable_drag_mode(result_label, drag_state, add_mode_ref, remove_mode_ref, select_mode_ref))
    select_btn = ttk.Button(bottom_buttons, text="Wybierz linie do Cobba",
        command=lambda: start_select_mode(result_label, select_mode_ref, selected_lines, add_mode_ref, remove_mode_ref, drag_state))

    for b in [add_green_btn, add_red_btn, remove_btn, move_btn, select_btn]:
        b.pack(side=tk.LEFT, padx=5)

    # Początkowe rysowanie — jeśli nie ma obrazu puste tło
    if current["original_img"] is None:
        # stwórz pusty obraz placeholder
        placeholder = Image.new("RGB", (800, 600), (20, 20, 20))
        current["original_img"] = placeholder

    tk_img, bg_img_id = redraw(
        canvas, current["original_img"], zoom_ref[0], offset_x_ref[0], offset_y_ref[0],
        current["pedicle_points"], selected_lines, point_items, lines_coords
    )
    canvas.image_ref = tk_img

    # Obsługa zmiany rozmiaru
    def on_resize(event):
        tk_img2, _ = redraw(
            canvas, current["original_img"], zoom_ref[0], offset_x_ref[0], offset_y_ref[0],
            current["pedicle_points"], selected_lines, point_items, lines_coords
        )
        canvas.image_ref = tk_img2
    canvas.bind("<Configure>", on_resize)
    # przerysowanie
    def handle_redraw(event=None):
        tk_img2, _ = redraw(
            canvas,
            current["original_img"],
            zoom_ref[0],
            offset_x_ref[0],
            offset_y_ref[0],
            current["pedicle_points"],
            selected_lines,
            point_items,
            lines_coords
        )
        if tk_img2 is not None:
            canvas.image_ref = tk_img2

    canvas.bind("<<Redraw>>", handle_redraw)

    # Bindowanie zdarzeń myszy
    canvas.bind("<Button-1>", lambda e: on_mouse_down(
        e, canvas, current["pedicle_points"], manual_points, point_items,
        add_mode_ref, remove_mode_ref, drag_state, offset_x_ref, offset_y_ref,
        current["original_img"], result_label, select_mode_ref, selected_lines,
        lines_coords, helper_items))
    canvas.bind("<B1-Motion>", lambda e: on_mouse_move(
        e, canvas, current["pedicle_points"], drag_state, offset_x_ref, offset_y_ref))
    canvas.bind("<ButtonRelease-1>", lambda e: on_mouse_up(
        e, canvas, current["pedicle_points"], drag_state, current["original_img"],
        zoom_ref, offset_x_ref, offset_y_ref))

    root.mainloop()
