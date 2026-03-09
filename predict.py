import cv2
import torch
import os
from facenet_pytorch import MTCNN
from AgeNet import *
from PIL import Image

test = "sample/img4.jpg"

device = "cuda" if torch.cuda.is_available() else "cpu"

mtcnn = MTCNN(keep_all=True, device=device)
model = AgeNet("models/best_model.pth")

os.makedirs("runs", exist_ok=True)

run_id = 0
while os.path.exists(f"runs/run_{run_id}"):
    run_id += 1

run_dir = f"runs/run_{run_id}"
os.makedirs(run_dir)

frame = cv2.imread(test)
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

boxes, probs = mtcnn.detect(rgb)

frame_draw = frame.copy()

if boxes is not None:
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)

        h, w, _ = rgb.shape

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        bw = x2 - x1
        bh = y2 - y1
        side = max(bw, bh)

        side = int(side * 1.2)

        nx1 = cx - side // 2
        ny1 = cy - side // 2
        nx2 = cx + side // 2
        ny2 = cy + side // 2

        nx1 = max(0, nx1)
        ny1 = max(0, ny1)
        nx2 = min(w, nx2)
        ny2 = min(h, ny2)

        face = rgb[ny1:ny2, nx1:nx2]
        if face.size == 0:
            continue

        image = face.copy()

        image = cv2.medianBlur(image, 3)

        blur = cv2.GaussianBlur(image, (0, 0), 3)
        image = cv2.addWeighted(image, 1.5, blur, -0.5, 0)

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(2.0, (8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        image = Image.fromarray(image)

        result = model.predict(image)

        gender = float(result["gender"][0][0])
        age = float(result["age"][0])

        gender_label = "F" if gender > 0.5 else "M"
        text = f"{gender_label} Age: {int(age)}"

        cv2.rectangle(frame_draw, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(
            frame_draw,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0,255,0),
            3
        )

        face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

        cv2.rectangle(face_bgr, (0,0), (face_bgr.shape[1]-1, face_bgr.shape[0]-1), (0,255,0), 2)
        cv2.putText(
            face_bgr,
            text,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0,255,0),
            3
        )

        cv2.imwrite(f"{run_dir}/face_{i}.png", face_bgr)

cv2.imwrite(f"{run_dir}/full_image.png", frame_draw)