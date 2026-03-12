import cv2
import time
from ultralytics import YOLO

MODEL_PATH = "best.pt"
CONF_THRESH = 0.35
MAX_FPS = 15

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Kamera açılamadı. Sistem kamera iznini kontrol et.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_time = 0
try:
    CAMERA_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    CAMERA_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        now = time.time()

        if now - prev_time < 1.0 / MAX_FPS:
            pass

        frame_time = now - prev_time
        calculated_fps = 1.0 / frame_time if frame_time > 0 else 0

        prev_time = now

        results = model.predict(frame, conf=CONF_THRESH, verbose=False)

        annotated_frame = results[0].plot()

        cv2.putText(annotated_frame,
                    f"CONF THRESHOLD: {CONF_THRESH:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(annotated_frame,
                    f"REAL-TIME FPS: {calculated_fps:.1f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        detections = results[0].boxes
        if detections:
            best_det = detections[0]

            x1, y1, x2, y2 = map(int, best_det.xyxy[0])
            box_center_x = (x1 + x2) // 2
            box_center_y = (y1 + y2) // 2

            confidence_score = best_det.conf.item()

            position_text = f"KONUM: X={box_center_x}, Y={box_center_y}"
            confidence_text = f"GUVEN SKORU: {confidence_score:.2f}"

            cv2.putText(annotated_frame,
                        position_text,
                        (10, CAMERA_HEIGHT - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(annotated_frame,
                        confidence_text,
                        (10, CAMERA_HEIGHT - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.circle(annotated_frame, (box_center_x, box_center_y), 5, (0, 0, 255), -1)

        else:
            cv2.putText(annotated_frame,
                        "YANGIN TESPITI YAPILAMADI",
                        (10, CAMERA_HEIGHT - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("YOLOv8 Fire Detection Metrics & Position", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()