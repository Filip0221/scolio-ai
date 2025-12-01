from ultralytics import YOLO
import numpy as np
import cv2

# Ładuje model i zwraca obraz z punktami
def load_model_and_points(model_path, image_path):
    model = YOLO(model_path)
    results = model.predict(image_path, imgsz=640)

    img_array = results[0].orig_img.copy()
    overlay = img_array.copy()
    pedicle_points = []

    # Wyciągniecie punktów z masek
    if results[0].masks is not None:
        for mask in results[0].masks.xy:
            mask = np.array(mask, dtype=np.int32)

            if cv2.contourArea(mask) < 100:
                continue

            cv2.fillPoly(overlay, [mask], color=(0, 255, 0))

            min_y = np.min(mask[:, 1])
            max_y = np.max(mask[:, 1])

            top_candidates = mask[mask[:, 1] == min_y]
            bottom_candidates = mask[mask[:, 1] == max_y]

            top_point = (float(np.mean(top_candidates[:, 0])), float(min_y))
            bottom_point = (float(np.mean(bottom_candidates[:, 0])), float(max_y))

            DUPLICATE_TOLERANCE_PX = 10

            duplicate_top = any(
                abs(x - top_point[0]) < DUPLICATE_TOLERANCE_PX and abs(y - top_point[1]) < DUPLICATE_TOLERANCE_PX and t == "top"
                for (x, y, _, t) in pedicle_points
            )
            duplicate_bottom = any(
                abs(x - bottom_point[0]) < DUPLICATE_TOLERANCE_PX and abs(y - bottom_point[1]) < DUPLICATE_TOLERANCE_PX and t == "bottom"
                for (x, y, _, t) in pedicle_points
            )

            if not duplicate_top:
                pedicle_points.append((top_point[0], top_point[1], "green", "top"))
            if not duplicate_bottom:
                pedicle_points.append((bottom_point[0], bottom_point[1], "red", "bottom"))

    alpha = 0.05
    img_array = cv2.addWeighted(overlay, alpha, img_array, 1 - alpha, 0)

    return img_array, pedicle_points
