from ultralytics import YOLO
import os
# Wczytaj najlepszy model po treningu
model = YOLO("runs/segment/pedicle_yolo11_retrain/weights/best.pt")

# Ewaluacja na zbiorze walidacyjnym
metrics = model.val()
print(metrics)

results = model.predict("images/test/images/20240402_141339_png.rf.8f1f310ebf34f62b67fa971d823c51f1.jpg", save=True, imgsz=640)
results[0].show()  # pokaże w oknie
