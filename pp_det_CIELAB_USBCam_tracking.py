import json
from pathlib import Path

import cv2
import numpy as np
from collections import deque


VIDEO_WINDOW = "Orange Ball Tracking (Lab)"
CONTROL_WINDOW = "Controls"
CONFIG_PATH = Path(__file__).with_name("pp_det_CIELAB_USBCam_tracking_controls.json")
DEFAULT_ROI = (30, 160, 600, 460)
MIN_CONFIRMED_HITS = 5
SCORE_CONFIRM_FRAMES = 10
MAX_JUMP_DISTANCE = 80
MAX_STEP_CHANGE = 60
USE_CIRCULARITY_FILTER = False
MIN_CIRCULARITY = 0.65
USE_FILL_RATIO_FILTER = False
MIN_FILL_RATIO = 0.45
MAX_FILL_RATIO = 1.15
TRACKBAR_SETTINGS = {
    "L_min": {"default": 0, "max": 255},
    "L_max": {"default": 255, "max": 255},
    "A_min": {"default": 130, "max": 255},
    "A_max": {"default": 200, "max": 255},
    "B_min": {"default": 130, "max": 255},
    "B_max": {"default": 255, "max": 255},
    "Trail": {"default": 32, "max": 120},
    "MaxDist": {"default": 80, "max": 250},
}


def nothing(x):
    pass


def get_default_control_values():
    return {name: setting["default"] for name, setting in TRACKBAR_SETTINGS.items()}


def sanitize_control_values(values):
    sanitized = get_default_control_values()

    if not isinstance(values, dict):
        return sanitized

    for name, setting in TRACKBAR_SETTINGS.items():
        try:
            value = int(values.get(name, sanitized[name]))
        except (TypeError, ValueError):
            value = sanitized[name]

        sanitized[name] = max(0, min(value, setting["max"]))

    return sanitized


def load_control_values():
    if not CONFIG_PATH.exists():
        return get_default_control_values()

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            values = json.load(config_file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"설정 파일을 읽을 수 없어 기본값을 사용합니다: {exc}")
        return get_default_control_values()

    print(f"저장된 컨트롤 값을 불러왔습니다: {CONFIG_PATH}")
    return sanitize_control_values(values)


def get_current_control_values():
    return {
        name: cv2.getTrackbarPos(name, CONTROL_WINDOW)
        for name in TRACKBAR_SETTINGS
    }


def apply_control_values(values):
    for name, value in sanitize_control_values(values).items():
        cv2.setTrackbarPos(name, CONTROL_WINDOW, value)


def save_control_values(values):
    values = sanitize_control_values(values)

    try:
        with CONFIG_PATH.open("w", encoding="utf-8") as config_file:
            json.dump(values, config_file, indent=2)
            config_file.write("\n")
    except OSError as exc:
        print(f"컨트롤 값을 저장하지 못했습니다: {exc}")
        return

    print(f"컨트롤 값을 저장했습니다: {CONFIG_PATH}")


class AdjustableRect:
    def __init__(self, min_size=30, edge_tolerance=10, reset_callback=None):
        self.rect = None
        self.frame_size = None
        self.min_size = min_size
        self.edge_tolerance = edge_tolerance
        self.reset_callback = reset_callback
        self.drag_mode = None
        self.last_mouse_pos = None

    def set_frame_shape(self, frame_shape):
        height, width = frame_shape[:2]
        self.frame_size = (width, height)
        if self.rect is None:
            self.rect = list(DEFAULT_ROI)
            self._clamp_rect()

    def _clamp_rect(self):
        if self.rect is None or self.frame_size is None:
            return

        width, height = self.frame_size
        x1, y1, x2, y2 = self.rect

        x1 = max(0, min(x1, width - self.min_size))
        y1 = max(0, min(y1, height - self.min_size))
        x2 = max(self.min_size, min(x2, width))
        y2 = max(self.min_size, min(y2, height))

        if x2 - x1 < self.min_size:
            if self.drag_mode == "left":
                x1 = x2 - self.min_size
            else:
                x2 = x1 + self.min_size

        if y2 - y1 < self.min_size:
            if self.drag_mode == "top":
                y1 = y2 - self.min_size
            else:
                y2 = y1 + self.min_size

        self.rect = [max(0, x1), max(0, y1), min(width, x2), min(height, y2)]

    def _hit_test(self, x, y):
        if self.rect is None:
            return None

        x1, y1, x2, y2 = self.rect
        tol = self.edge_tolerance
        on_horizontal = x1 - tol <= x <= x2 + tol
        on_vertical = y1 - tol <= y <= y2 + tol

        if on_vertical and abs(x - x1) <= tol:
            return "left"
        if on_vertical and abs(x - x2) <= tol:
            return "right"
        if on_horizontal and abs(y - y1) <= tol:
            return "top"
        if on_horizontal and abs(y - y2) <= tol:
            return "bottom"

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        center_w = max(30, min(120, (x2 - x1) // 3))
        center_h = max(30, min(120, (y2 - y1) // 3))
        if abs(x - center_x) <= center_w // 2 and abs(y - center_y) <= center_h // 2:
            return "move"

        return None

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MBUTTONDOWN:
            if self.reset_callback:
                self.reset_callback()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_mode = self._hit_test(x, y)
            self.last_mouse_pos = (x, y)
            return

        if event == cv2.EVENT_MOUSEMOVE and self.drag_mode and (flags & cv2.EVENT_FLAG_LBUTTON):
            last_x, last_y = self.last_mouse_pos
            dx = x - last_x
            dy = y - last_y
            x1, y1, x2, y2 = self.rect

            if self.drag_mode == "move":
                x1 += dx
                x2 += dx
                y1 += dy
                y2 += dy
                width, height = self.frame_size
                if x1 < 0:
                    x2 -= x1
                    x1 = 0
                if y1 < 0:
                    y2 -= y1
                    y1 = 0
                if x2 > width:
                    x1 -= x2 - width
                    x2 = width
                if y2 > height:
                    y1 -= y2 - height
                    y2 = height
            elif self.drag_mode == "left":
                x1 += dx
            elif self.drag_mode == "right":
                x2 += dx
            elif self.drag_mode == "top":
                y1 += dy
            elif self.drag_mode == "bottom":
                y2 += dy

            self.rect = [x1, y1, x2, y2]
            self._clamp_rect()
            self.last_mouse_pos = (x, y)
            return

        if event == cv2.EVENT_LBUTTONUP:
            self.drag_mode = None
            self.last_mouse_pos = None

    def draw(self, frame):
        if self.rect is None:
            return

        x1, y1, x2, y2 = self.rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 80), 2)

    def contains_point(self, point):
        if self.rect is None:
            return False

        x, y = point
        x1, y1, x2, y2 = self.rect
        return x1 <= x <= x2 and y1 <= y <= y2


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
            "recent_steps": deque(maxlen=6),
            "hits": 1,
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
            dx = detection["center"][0] - track["center"][0]
            dy = detection["center"][1] - track["center"][1]
            track["recent_steps"].append((dx * dx + dy * dy) ** 0.5)
            track["center"] = detection["center"]
            track["radius"] = detection["radius"]
            track["history"].append(detection["center"])
            track["hits"] += 1
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


def is_reliable_track(track):
    if track["missed"] > 0 or track["hits"] < MIN_CONFIRMED_HITS:
        return False

    recent_steps = list(track["recent_steps"])
    if not recent_steps:
        return False

    if max(recent_steps) > MAX_JUMP_DISTANCE:
        return False

    for prev_step, curr_step in zip(recent_steps, recent_steps[1:]):
        if abs(curr_step - prev_step) > MAX_STEP_CHANGE:
            return False

    return True


class GoalScoreCounter:
    def __init__(self, confirm_frames=SCORE_CONFIRM_FRAMES):
        self.confirm_frames = confirm_frames
        self.score = 0
        self.inside_states = {}

    def reset(self):
        self.score = 0
        self.inside_states.clear()
        print("Score reset: 0")

    def update(self, inside_track_ids):
        inside_track_ids = set(inside_track_ids)
        next_states = {}

        for track_id in inside_track_ids:
            state = self.inside_states.get(track_id, {"frames": 0, "scored": False})
            state["frames"] += 1

            if state["frames"] >= self.confirm_frames and not state["scored"]:
                self.score += 1
                state["scored"] = True
                print(f"GOAL! score={self.score}, ID={track_id}")

            next_states[track_id] = state

        self.inside_states = next_states

    def draw(self, frame):
        text = str(self.score)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 2.0
        thickness = 4
        margin = 24
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        x = frame.shape[1] - text_size[0] - margin
        y = margin + text_size[1]

        cv2.putText(frame, text, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 2)
        cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness)


def create_controls_window():
    control_values = load_control_values()

    cv2.namedWindow(CONTROL_WINDOW, cv2.WINDOW_NORMAL)
    for name, setting in TRACKBAR_SETTINGS.items():
        cv2.createTrackbar(
            name,
            CONTROL_WINDOW,
            control_values[name],
            setting["max"],
            nothing,
        )


def main():
    create_controls_window()
    score_counter = GoalScoreCounter()
    roi_rect = AdjustableRect(reset_callback=score_counter.reset)
    cv2.namedWindow(VIDEO_WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(VIDEO_WINDOW, roi_rect.on_mouse)

    # 0번 장치는 일반적으로 첫 번째 웹캠 (/dev/video0)
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return 1

    print("카메라 추적을 시작합니다. s=컨트롤 값 저장, a=초기값 복원, ESC 또는 Ctrl+C=종료")
    tracker = BallTracker()
    printed_frame_size = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break

            frame = cv2.flip(frame, 1)

            if not printed_frame_size:
                height, width = frame.shape[:2]
                print(f"카메라 프레임 크기: {width}x{height}")
                printed_frame_size = True

            roi_rect.set_frame_shape(frame.shape)

            blurred = cv2.GaussianBlur(frame, (11, 11), 0)

            # BGR -> CIELAB 변환
            lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2Lab)

            # 트랙바에서 현재 임계값 읽기
            L_min = cv2.getTrackbarPos('L_min', CONTROL_WINDOW)
            L_max = cv2.getTrackbarPos('L_max', CONTROL_WINDOW)
            A_min = cv2.getTrackbarPos('A_min', CONTROL_WINDOW)
            A_max = cv2.getTrackbarPos('A_max', CONTROL_WINDOW)
            B_min = cv2.getTrackbarPos('B_min', CONTROL_WINDOW)
            B_max = cv2.getTrackbarPos('B_max', CONTROL_WINDOW)
            trail_length = max(2, cv2.getTrackbarPos('Trail', CONTROL_WINDOW))
            max_distance = max(10, cv2.getTrackbarPos('MaxDist', CONTROL_WINDOW))

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
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter <= 0:
                        continue

                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if USE_CIRCULARITY_FILTER and circularity < MIN_CIRCULARITY:
                        continue

                    ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                    if radius > 5:
                        circle_area = np.pi * radius * radius
                        fill_ratio = area / circle_area
                        if USE_FILL_RATIO_FILTER and not MIN_FILL_RATIO <= fill_ratio <= MAX_FILL_RATIO:
                            continue

                        detections.append({
                            "center": (int(x), int(y)),
                            "radius": int(radius),
                            "area": area,
                            "circularity": circularity,
                            "fill_ratio": fill_ratio,
                        })

            tracks = tracker.update(detections)
            inside_track_ids = []

            for track_id, track in tracks.items():
                if not is_reliable_track(track):
                    continue

                center = track["center"]
                radius = track["radius"]
                history = list(track["history"])
                is_inside_roi = roi_rect.contains_point(center)
                draw_color = (80, 255, 80) if is_inside_roi else (0, 255, 255)

                cv2.circle(frame, center, radius, draw_color, 2)
                cv2.putText(frame, f"ID {track_id} ({center[0]}, {center[1]})",
                            (center[0] - 60, center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                for i in range(1, len(history)):
                    thickness = int(np.interp(i, [1, len(history) - 1], [1, 4]))
                    cv2.line(frame, history[i - 1], history[i], draw_color, thickness)

                if is_inside_roi:
                    inside_track_ids.append(track_id)
                    print(f"ROI ID {track_id}: center={center}, radius={radius}, hits={track['hits']}")

            score_counter.update(inside_track_ids)
            roi_rect.draw(frame)
            score_counter.draw(frame)

            cv2.imshow(VIDEO_WINDOW, frame)
            # cv2.imshow("Mask", mask)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == ord('s'):
                save_control_values(get_current_control_values())
            elif key == ord('a'):
                apply_control_values(get_default_control_values())
                print("컨트롤 값을 초기값으로 되돌렸습니다.")
    except KeyboardInterrupt:
        print("\n사용자 중단으로 종료합니다.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
