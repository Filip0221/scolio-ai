# Punkty na obrazie (automatyczne i ręczne)
pedicle_points = []
manual_points = []

# Parametry widoku
zoom = 1.0
ZOOM_MIN, ZOOM_MAX = 0.2, 6.0
offset_x, offset_y = 0, 0

# Obiekty graficzne Tkinter
tk_img = None
bg_img_id = None

# Elementy interakcji użytkownika
point_items = {}
add_mode = None
drag_point_id = None
drag_point_idx = None
is_panning = False
pan_start = (0, 0)
offset_start = (0, 0)

# Dane pomocnicze dla linii
lines_coords = []
select_mode = False
selected_lines = []
helper_items = []

# Stałe rysowania
POINT_RADIUS = 3
PAIR_Y_TOL = 30
LINE_EXTEND = 100

# Referencje do głównych widżetów
root = None
canvas = None
result_label = None
original_img = None
