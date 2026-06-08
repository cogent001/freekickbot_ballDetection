# -*- coding: utf-8 -*-
import sys

import cv2
import numpy as np
from PyQt5.QtCore import QEvent, QTimer, Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QLabel

from UI import (
    FONT_EMOJI,
    FONT_SANS,
    ICO_WAIT,
    MainWindow,
    TEXT_GRAY,
)
from pp_det_CIELAB_USBCam_tracking import (
    CONTROL_WINDOW,
    MIN_CIRCULARITY,
    MIN_FILL_RATIO,
    MAX_FILL_RATIO,
    USE_CIRCULARITY_FILTER,
    USE_FILL_RATIO_FILTER,
    AdjustableRect,
    BallTracker,
    GoalScoreCounter,
    apply_control_values,
    create_controls_window,
    get_current_control_values,
    get_default_control_values,
    is_reliable_track,
    save_control_values,
)


class TrackingProcessor:
    def __init__(self):
        self.controls_ready = False
        self.score_counter = GoalScoreCounter()
        self.roi_rect = AdjustableRect(reset_callback=self.reset_score)
        self.tracker = BallTracker()
        self.last_frame_shape = None
        self._create_controls()

    def _create_controls(self):
        if self.controls_ready:
            return

        create_controls_window()
        self.controls_ready = True

    def reset_score(self):
        self.score_counter.reset()

    def _get_trackbar(self, name, default):
        try:
            return cv2.getTrackbarPos(name, CONTROL_WINDOW)
        except cv2.error:
            self.controls_ready = False
            self._create_controls()
            return default

    def _handle_controls_window_events(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            save_control_values(get_current_control_values())
        elif key == ord("a"):
            apply_control_values(get_default_control_values())
            print("Control values restored to defaults.")

    def process(self, frame, scoring_enabled=True):
        self._create_controls()
        self._handle_controls_window_events()

        frame = cv2.flip(frame, 1)
        self.last_frame_shape = frame.shape
        self.roi_rect.set_frame_shape(frame.shape)

        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2Lab)

        l_min = self._get_trackbar("L_min", 0)
        l_max = self._get_trackbar("L_max", 255)
        a_min = self._get_trackbar("A_min", 130)
        a_max = self._get_trackbar("A_max", 200)
        b_min = self._get_trackbar("B_min", 130)
        b_max = self._get_trackbar("B_max", 255)
        trail_length = max(2, self._get_trackbar("Trail", 32))
        max_distance = max(10, self._get_trackbar("MaxDist", 80))

        self.tracker.max_distance = max_distance
        self.tracker.set_trail_length(trail_length)

        lower_lab = np.array([l_min, a_min, b_min])
        upper_lab = np.array([l_max, a_max, b_max])
        mask = cv2.inRange(lab, lower_lab, upper_lab)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= 50:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if USE_CIRCULARITY_FILTER and circularity < MIN_CIRCULARITY:
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius <= 5:
                continue

            circle_area = np.pi * radius * radius
            fill_ratio = area / circle_area
            if USE_FILL_RATIO_FILTER and not MIN_FILL_RATIO <= fill_ratio <= MAX_FILL_RATIO:
                continue

            detections.append(
                {
                    "center": (int(x), int(y)),
                    "radius": int(radius),
                    "area": area,
                    "circularity": circularity,
                    "fill_ratio": fill_ratio,
                }
            )

        tracks = self.tracker.update(detections)
        inside_track_ids = []

        for track_id, track in tracks.items():
            if not is_reliable_track(track):
                continue

            center = track["center"]
            radius = track["radius"]
            history = list(track["history"])
            is_inside_roi = self.roi_rect.contains_point(center)
            draw_color = (80, 255, 80) if is_inside_roi else (0, 255, 255)

            cv2.circle(frame, center, radius, draw_color, 2)
            cv2.putText(
                frame,
                "ID %d (%d, %d)" % (track_id, center[0], center[1]),
                (center[0] - 60, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            for i in range(1, len(history)):
                thickness = int(np.interp(i, [1, len(history) - 1], [1, 4]))
                cv2.line(frame, history[i - 1], history[i], draw_color, thickness)

            if is_inside_roi:
                inside_track_ids.append(track_id)

        if scoring_enabled:
            self.score_counter.update(inside_track_ids)

        self.roi_rect.draw(frame)
        self.score_counter.draw(frame)
        return frame, self.score_counter.score

    def handle_mouse_event(self, cv_event, x, y, flags=0):
        if self.last_frame_shape is None:
            return

        self.roi_rect.set_frame_shape(self.last_frame_shape)
        self.roi_rect.on_mouse(cv_event, x, y, flags, None)

    def close(self):
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


class TrackingMainWindow(MainWindow):
    SCORE_FONT_SIZE = 208

    def __init__(self):
        self.processor = TrackingProcessor()
        self._score_flash_on = False
        self._goal_pulse_index = 0
        super().__init__()
        self._adjust_game_layout()
        self._setup_waiting_prompt()
        self._setup_score_activity_timer()
        self._setup_goal_effects()
        self._set_waiting_score_style()
        self.camera_panel.video_label.setMouseTracking(True)
        self.camera_panel.video_label.installEventFilter(self)

    def _adjust_game_layout(self):
        self.timer_panel.setFixedHeight(104)
        self.status_panel.setFixedHeight(70)

        self.timer_panel.time_label.setFont(QFont(FONT_SANS, 42, QFont.Bold))
        self.score_panel.score_label.setFont(QFont(FONT_SANS, self.SCORE_FONT_SIZE, QFont.Bold))

        timer_layout = self.timer_panel.layout()
        if timer_layout is not None:
            timer_layout.setContentsMargins(22, 12, 22, 12)

        for child in self.timer_panel.findChildren(type(self.timer_panel.time_label)):
            if child is self.timer_panel.time_label:
                continue
            text = child.text()
            if text == "\uc2dc\uac04":
                child.setFont(QFont(FONT_SANS, 24, QFont.Bold))
            elif text:
                child.setFont(QFont(FONT_EMOJI, 22))

    def _setup_waiting_prompt(self):
        self.waiting_prompt_label = QLabel("START\ubc84\ud2bc \ub20c\ub7ec\uc8fc\uc138\uc694!")
        self.waiting_prompt_label.setAlignment(Qt.AlignCenter)
        self.waiting_prompt_label.setFont(QFont(FONT_SANS, 24, QFont.Bold))
        self.waiting_prompt_label.setStyleSheet(
            "color:#ffd56a;background:transparent;border:none;padding-bottom:18px;"
        )

        score_layout = self.score_panel.layout()
        if score_layout is not None:
            score_layout.addWidget(self.waiting_prompt_label)

    def _setup_score_activity_timer(self):
        self.score_activity_timer = QTimer(self)
        self.score_activity_timer.timeout.connect(self._toggle_active_score_style)

    def _setup_goal_effects(self):
        self.goal_label = QLabel("GOAL!", self.score_panel)
        self.goal_label.setAlignment(Qt.AlignCenter)
        self.goal_label.setFont(QFont(FONT_SANS, 38, QFont.Bold))
        self.goal_label.setStyleSheet(
            "color:#ffffff;background:rgba(0,216,112,185);"
            "border:2px solid #ffffff;border-radius:12px;padding:4px 18px;"
        )
        self.goal_label.hide()

        self.goal_hide_timer = QTimer(self)
        self.goal_hide_timer.setSingleShot(True)
        self.goal_hide_timer.timeout.connect(self.goal_label.hide)

        self.goal_pulse_timer = QTimer(self)
        self.goal_pulse_timer.setInterval(70)
        self.goal_pulse_timer.timeout.connect(self._advance_goal_pulse)
        self.goal_pulse_steps = [
            (260, "#ffffff"),
            (244, "#fff0a8"),
            (228, "#ffffff"),
            (216, "#d6ffe7"),
            (self.SCORE_FONT_SIZE, "#b8ffd6"),
        ]

    def _position_goal_label(self):
        width = min(260, max(190, self.score_panel.width() - 80))
        height = 64
        x = (self.score_panel.width() - width) // 2
        y = 58
        self.goal_label.setGeometry(x, y, width, height)

    def _set_waiting_score_style(self):
        self._score_flash_on = False
        self.score_panel.setStyleSheet(
            "ScorePanel{background:#0d1e35;border:2px solid #f5a800;border-radius:14px;}"
        )
        self.score_panel.score_label.setStyleSheet(
            "color:#f5a800;background:transparent;border:none;"
        )

    def _set_active_score_style(self, bright=False):
        bg_color = "#12352c" if bright else "#0f2a28"
        border_color = "#00ff88" if bright else "#00d870"
        score_color = "#f7fff9" if bright else "#b8ffd6"
        self.score_panel.setStyleSheet(
            "ScorePanel{background:%s;border:3px solid %s;border-radius:14px;}"
            % (bg_color, border_color)
        )
        self.score_panel.score_label.setStyleSheet(
            "color:%s;background:transparent;border:none;" % score_color
        )

    def _toggle_active_score_style(self):
        if not self.running:
            self.score_activity_timer.stop()
            self._set_waiting_score_style()
            return

        self._score_flash_on = not self._score_flash_on
        self._set_active_score_style(self._score_flash_on)

    def _start_score_activity(self):
        self._score_flash_on = True
        self._set_active_score_style(True)
        self.score_activity_timer.start(500)

    def _stop_score_activity(self):
        self.score_activity_timer.stop()
        self.goal_hide_timer.stop()
        self.goal_pulse_timer.stop()
        self.goal_label.hide()
        self.score_panel.score_label.setFont(QFont(FONT_SANS, self.SCORE_FONT_SIZE, QFont.Bold))
        self.waiting_prompt_label.show()
        self._set_waiting_score_style()

    def _play_goal_effect(self):
        self.score_activity_timer.stop()
        self._set_active_score_style(True)
        self._position_goal_label()
        self.goal_label.raise_()
        self.goal_label.show()
        self.goal_hide_timer.start(900)

        self._goal_pulse_index = 0
        self.goal_pulse_timer.start()
        self._advance_goal_pulse()

    def _advance_goal_pulse(self):
        if self._goal_pulse_index >= len(self.goal_pulse_steps):
            self.goal_pulse_timer.stop()
            self.score_panel.score_label.setFont(QFont(FONT_SANS, self.SCORE_FONT_SIZE, QFont.Bold))
            if self.running:
                self._start_score_activity()
            else:
                self._set_waiting_score_style()
            return

        font_size, color = self.goal_pulse_steps[self._goal_pulse_index]
        self.score_panel.score_label.setFont(QFont(FONT_SANS, font_size, QFont.Bold))
        self.score_panel.score_label.setStyleSheet(
            "color:%s;background:transparent;border:none;" % color
        )
        self._goal_pulse_index += 1

    def _setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            self.camera_panel.show_no_signal()
            self.cap = None

    def _update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.camera_panel.show_no_signal()
            return

        processed_frame, detected_score = self.processor.process(
            frame,
            scoring_enabled=self.running,
        )
        self.camera_panel.update_frame(processed_frame)

        if self.running and detected_score != self.score:
            scored = detected_score > self.score
            self.score = detected_score
            self.score_panel.set_score(self.score)
            if scored:
                self._play_goal_effect()

    def _start_game(self):
        self.processor.reset_score()
        self.waiting_prompt_label.hide()
        super()._start_game()
        self._start_score_activity()

    def _end_game(self):
        self.running = False
        self.game_timer.stop()
        self._stop_score_activity()
        self.status_panel.set_status("\ub300\uae30", ICO_WAIT, TEXT_GRAY)
        self.status_panel.set_running(False)

    def _label_pos_to_frame_pos(self, pos):
        frame_shape = self.processor.last_frame_shape
        if frame_shape is None:
            return None

        frame_h, frame_w = frame_shape[:2]
        label = self.camera_panel.video_label
        label_w = label.width()
        label_h = label.height()
        if frame_w <= 0 or frame_h <= 0 or label_w <= 0 or label_h <= 0:
            return None

        scale = min(label_w / float(frame_w), label_h / float(frame_h))
        draw_w = int(frame_w * scale)
        draw_h = int(frame_h * scale)
        offset_x = (label_w - draw_w) // 2
        offset_y = (label_h - draw_h) // 2

        x = pos.x() - offset_x
        y = pos.y() - offset_y
        if x < 0 or y < 0 or x >= draw_w or y >= draw_h:
            return None

        frame_x = int(x / scale)
        frame_y = int(y / scale)
        frame_x = max(0, min(frame_w - 1, frame_x))
        frame_y = max(0, min(frame_h - 1, frame_y))
        return frame_x, frame_y

    def eventFilter(self, watched, event):
        if watched is self.camera_panel.video_label:
            if event.type() in (
                QEvent.MouseButtonPress,
                QEvent.MouseMove,
                QEvent.MouseButtonRelease,
            ):
                frame_pos = self._label_pos_to_frame_pos(event.pos())
                if frame_pos is None:
                    return False

                x, y = frame_pos
                flags = cv2.EVENT_FLAG_LBUTTON if event.buttons() & Qt.LeftButton else 0

                if event.type() == QEvent.MouseButtonPress:
                    if event.button() == Qt.LeftButton:
                        self.processor.handle_mouse_event(cv2.EVENT_LBUTTONDOWN, x, y, flags)
                        return True
                    if event.button() == Qt.MiddleButton:
                        self.processor.handle_mouse_event(cv2.EVENT_MBUTTONDOWN, x, y, flags)
                        self.score = 0
                        self.score_panel.set_score(0)
                        return True

                if event.type() == QEvent.MouseMove:
                    self.processor.handle_mouse_event(cv2.EVENT_MOUSEMOVE, x, y, flags)
                    return bool(flags)

                if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                    self.processor.handle_mouse_event(cv2.EVENT_LBUTTONUP, x, y, flags)
                    return True

        return super().eventFilter(watched, event)

    def showEvent(self, event):
        super().showEvent(event)
        self._position_goal_label()
        self.raise_()
        self.activateWindow()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "goal_label"):
            self._position_goal_label()

    def closeEvent(self, event):
        self._stop_score_activity()
        self.processor.close()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = TrackingMainWindow()
    window.show()
    window.raise_()
    window.activateWindow()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
