from ultralytics import YOLO
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

model = YOLO("yolo11n.pt")

results = model.train(
    data="images/data.yaml",
    epochs=100,            # dłużej z early stopping
    imgsz=640,
    batch=4,               
    lr0=0.0005,            # wolniejsze, dokładniejsze uczenie
    patience=15,           # zatrzyma się automatycznie
    device="mps",
    augment=True,          # rotacje, flip, jasność, zoom
    name="pedicle_yolo11_retrain"
)
yolo biblioteki zbior danych podzial danych augumentacja danych