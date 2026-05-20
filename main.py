from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

model = YOLO("runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture("http://192.168.1.3:8080/video")  # Nhớ thay id nhé
line_y = 250

counted_ids = set()
error_ids = set()

coca_ok = 0
error_count = 0

font = ImageFont.truetype("arial.ttf", 24)


def draw_text(frame, text, position, color=(255, 0, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


warning_timer = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    results = model.track(frame, persist=True, conf=0.5)

    just_detected_error = False

    if results and results[0].boxes.id is not None:
        boxes = results[0].boxes

        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            obj_id = int(boxes.id[i])

            cls = int(boxes.cls[i])
            label = model.names[cls].lower()

            center_y = (y1 + y2) // 2

            is_error = False

            if center_y > line_y and obj_id not in counted_ids:
                counted_ids.add(obj_id)

                if label == "coca":
                    coca_ok += 1

                elif label == "coca_open":
                    error_count += 1
                    error_ids.add(obj_id)
                    just_detected_error = True

            if label == "coca":
                color = (0, 255, 0)  # xanh lá

            elif label == "coca_open":
                color = (0, 0, 255)  # đỏ

            else:
                color = (255, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            status = "TỐT" if not is_error else "LỖI"
            text = f"{label} | {status} | ID:{obj_id}"

            frame = draw_text(frame, text, (x1, y1 - 25), color=color[::-1])

    cv2.line(frame, (0, line_y), (640, line_y), (255, 255, 0), 2)

    total = coca_ok + error_count

    cv2.putText(frame, f"Coca dat: {coca_ok}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Loi (lon mo): {error_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, f"Tong: {total}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if just_detected_error:
        warning_timer = 30

    if warning_timer > 0:
        frame = draw_text(
            frame,
            f"CẢNH BÁO: {len(error_ids)} LON BỊ MỞ",
            (10, 420),
            color=(255, 0, 0)
        )
        warning_timer -= 1

    cv2.imshow("HE THONG KIEM TRA COCA", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
