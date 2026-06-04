import cv2
import numpy as np
from collections import deque


def nothing(x):
    pass


class BallTracker:
    def __init__(self, max_distance=80, max_missed=8, trail_length=32):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.trail_length = trail_length
        self.tracks = {}
        self.next_id = 1

    def _new_track(self, detection):
        track_id = self.next_id
        self.next_id += 1
        self.tracks[track_id] = {
            "center": detection["center"],
            "radius": detection["radius"],
            "history": deque([detection["center"]], maxlen=self.trail_length),
            "missed": 0,
        }

    def update(self, detections):
        unmatched_detections = set(range(len(detections)))

        for track_id, track in list(self.tracks.items()):
            best_idx = None
            best_dist = self.max_distance
            tx, ty = track["center"]

            for idx in unmatched_detections:
                dx = detections[idx]["center"][0] - tx
                dy = detections[idx]["center"][1] - ty
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx is None:
                track["missed"] += 1
                if track["missed"] > self.max_missed:
                    del self.tracks[track_id]
                continue

            detection = detections[best_idx]
            unmatched_detections.remove(best_idx)
            track["center"] = detection["center"]
            track["radius"] = detection["radius"]
            track["history"].append(detection["center"])
            track["missed"] = 0

        for idx in unmatched_detections:
            self._new_track(detections[idx])

        return self.tracks

    def set_trail_length(self, trail_length):
        if trail_length == self.trail_length:
            return

        self.trail_length = trail_length
        for track in self.tracks.values():
            track["history"] = deque(track["history"], maxlen=self.trail_length)


cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
cv2.createTrackbar('L_min', 'Controls', 0, 255, nothing)
cv2.createTrackbar('L_max', 'Controls', 255, 255, nothing)
cv2.createTrackbar('A_min', 'Controls', 130, 255, nothing)
cv2.createTrackbar('A_max', 'Controls', 200, 255, nothing)
cv2.createTrackbar('B_min', 'Controls', 130, 255, nothing)
cv2.createTrackbar('B_max', 'Controls', 255, 255, nothing)
cv2.createTrackbar('Trail', 'Controls', 32, 120, nothing)
cv2.createTrackbar('MaxDist', 'Controls', 80, 250, nothing)

# 0번 장치는 일반적으로 첫 번째 웹캠 (/dev/video0)
cap = cv2.VideoCapture(0)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

tracker = BallTracker()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    # BGR -> CIELAB 변환
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2Lab)

    # 트랙바에서 현재 임계값 읽기
    L_min = cv2.getTrackbarPos('L_min', 'Controls')
    L_max = cv2.getTrackbarPos('L_max', 'Controls')
    A_min = cv2.getTrackbarPos('A_min', 'Controls')
    A_max = cv2.getTrackbarPos('A_max', 'Controls')
    B_min = cv2.getTrackbarPos('B_min', 'Controls')
    B_max = cv2.getTrackbarPos('B_max', 'Controls')
    trail_length = max(2, cv2.getTrackbarPos('Trail', 'Controls'))
    max_distance = max(10, cv2.getTrackbarPos('MaxDist', 'Controls'))

    tracker.max_distance = max_distance
    tracker.set_trail_length(trail_length)

    lower_lab = np.array([L_min, A_min, B_min])
    upper_lab = np.array([L_max, A_max, B_max])

    mask = cv2.inRange(lab, lower_lab, upper_lab)

    # 모폴로지 연산으로 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 컨투어 검출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:  # 잡음 제거용 최소 면적
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            if radius > 5:
                detections.append({
                    "center": (int(x), int(y)),
                    "radius": int(radius),
                    "area": area,
                })

    tracks = tracker.update(detections)

    for track_id, track in tracks.items():
        if track["missed"] > 0:
            continue

        center = track["center"]
        radius = track["radius"]
        history = list(track["history"])

        cv2.circle(frame, center, radius, (0, 255, 255), 2)
        cv2.putText(frame, f"ID {track_id} ({center[0]}, {center[1]})",
                    (center[0] - 60, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for i in range(1, len(history)):
            thickness = int(np.interp(i, [1, len(history) - 1], [1, 4]))
            cv2.line(frame, history[i - 1], history[i], (0, 180, 255), thickness)

        print(f"ID {track_id}: center={center}, radius={radius}")

    cv2.imshow("Orange Ball Tracking (Lab)", frame)
    # cv2.imshow("Mask", mask)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
