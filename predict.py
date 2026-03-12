import argparse
import cv2
import torch
import os
import sys
import logging
from datetime import datetime
from facenet_pytorch import MTCNN
from AgeNet import *
from PIL import Image
import AgeNet

VERSION = AgeNet.__version__


def setup_logger(level, log_dir):
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

    logger = logging.getLogger("agenet_cli")
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


def create_run_dir(base):
    os.makedirs(base, exist_ok=True)

    run_id = 0
    while os.path.exists(f"{base}/run_{run_id}"):
        run_id += 1

    run_dir = f"{base}/run_{run_id}"
    os.makedirs(run_dir)

    return run_dir


def preprocess_face(face):
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

    return Image.fromarray(image)


def crop_with_padding(box, rgb, padding):
    x1, y1, x2, y2 = box.astype(int)

    h, w, _ = rgb.shape

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    bw = x2 - x1
    bh = y2 - y1
    side = max(bw, bh)

    side = int(side * (1 + padding))

    nx1 = cx - side // 2
    ny1 = cy - side // 2
    nx2 = cx + side // 2
    ny2 = cy + side // 2

    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(w, nx2)
    ny2 = min(h, ny2)

    return rgb[ny1:ny2, nx1:nx2], (x1, y1, x2, y2)


def process_frame(frame, mtcnn, model, args, run_dir, logger):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, probs = mtcnn.detect(rgb)

    frame_draw = frame.copy()

    if boxes is None:
        logger.info("No face detected")
        return frame_draw

    for i, box in enumerate(boxes):

        face, (x1, y1, x2, y2) = crop_with_padding(box, rgb, args.padding)

        if face.size == 0:
            continue

        image = preprocess_face(face)

        result = model.predict(image)

        gender = float(result["gender"][0][0])
        age = float(result["age"][0])

        gender_label = "F" if gender > 0.5 else "M"

        text = f"{gender_label} Age: {int(age)}"

        logger.info(f"Face {i}: {text}")

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

        if args.crop:

            face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

            cv2.rectangle(
                face_bgr,
                (0,0),
                (face_bgr.shape[1]-1, face_bgr.shape[0]-1),
                (0,255,0),
                2
            )

            cv2.putText(
                face_bgr,
                text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0,255,0),
                3
            )

            if args.save:
                cv2.imwrite(f"{run_dir}/face_{i}.png", face_bgr)

    return frame_draw


def run_image(args, mtcnn, model, run_dir, logger):

    frame = cv2.imread(args.image)

    if frame is None:
        logger.error("Image not found")
        return

    result = process_frame(frame, mtcnn, model, args, run_dir, logger)

    if args.save:
        cv2.imwrite(f"{run_dir}/full_image.png", result)

    if args.imshow:
        cv2.imshow("result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_camera(args, mtcnn, model, run_dir, logger):

    cap = cv2.VideoCapture(args.camera)

    while True:

        ret, frame = cap.read()

        if not ret:
            logger.error("Camera read failed")
            break

        result = process_frame(frame, mtcnn, model, args, run_dir, logger)

        if args.imshow:
            cv2.imshow("camera", result)

        if args.save:
            cv2.imwrite(f"{run_dir}/frame_{int(datetime.now().timestamp())}.png", result)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():

    parser = argparse.ArgumentParser(
        description="AgeNet Face Age/Gender CLI"
    )

    parser.add_argument(
        "--image",
        type=str,
        help="Path to image"
    )

    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Camera index or rtsp url"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu","cuda"]
    )

    parser.add_argument(
        "--padding",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--save",
        action="store_true"
    )

    parser.add_argument(
        "--save-path",
        default="runs"
    )

    parser.add_argument(
        "--crop",
        action="store_true"
    )

    parser.add_argument(
        "--imshow",
        action="store_true"
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG","INFO","WARNING","ERROR"]
    )

    parser.add_argument(
        "--version",
        action="store_true"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    if args.version:
        print(VERSION)
        return

    device = args.device

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = create_run_dir(args.save_path)

    logger = setup_logger(args.log_level, run_dir)

    logger.info(f"Version: {VERSION}")
    logger.info(f"Device: {device}")

    mtcnn = MTCNN(keep_all=True, device=device)

    model = AgeNet("models/best_model.pth")

    if args.image:
        run_image(args, mtcnn, model, run_dir, logger)

    elif args.camera is not None:

        try:
            cam = int(args.camera)
        except:
            cam = args.camera

        args.camera = cam

        run_camera(args, mtcnn, model, run_dir, logger)

    else:
        logger.error("Specify --image or --camera")


if __name__ == "__main__":
    main()