# -*- coding: utf-8 -*-
import sys
import platform
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont

_IS_WINDOWS = platform.system() == "Windows"
FONT_SANS  = "Arial"          if _IS_WINDOWS else "Liberation Sans"
FONT_EMOJI = "Segoe UI Emoji" if _IS_WINDOWS else "Noto Color Emoji"

BG_COLOR     = "#0a1628"
PANEL_BG     = "#0d1e35"
BORDER_CYAN  = "#00c8ff"
BORDER_YELLOW= "#f5a800"
TEXT_CYAN    = "#00c8ff"
TEXT_YELLOW  = "#f5a800"
TEXT_GRAY    = "#7a9ab5"
BTN_BG       = "#0d3a5c"
BTN_HOVER    = "#1a5a8a"

ICO_CAM   = "\U0001f4f7"   # camera
ICO_CLOCK = "\U0001f550"   # clock
ICO_STAR  = "\u2605"       # star
ICO_WAIT  = "\u231b"       # hourglass
ICO_FLAG  = "\U0001f3c1"   # flag


class RoundedPanel(QFrame):
    def __init__(self, border=BORDER_CYAN, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            "RoundedPanel{background:%s;border:2px solid %s;border-radius:14px;}"
            % (PANEL_BG, border)
        )


class CameraPanel(RoundedPanel):
    def __init__(self, parent=None):
        super().__init__(BORDER_CYAN, parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 12)
        lay.setSpacing(8)

        hdr = QWidget()
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(0,0,0,0); hl.setSpacing(6)
        ico = QLabel(ICO_CAM)
        ico.setFont(QFont(FONT_EMOJI, 11))
        ico.setStyleSheet("color:%s;background:transparent;border:none;" % TEXT_CYAN)
        lbl = QLabel("CAM")
        lbl.setFont(QFont(FONT_SANS, 11, QFont.Bold))
        lbl.setStyleSheet(
            "color:%s;background:transparent;border:1px solid %s;"
            "border-radius:6px;padding:2px 10px;" % (TEXT_CYAN, BORDER_CYAN)
        )
        hl.addWidget(ico); hl.addWidget(lbl); hl.addStretch()
        lay.addWidget(hdr)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#000;border-radius:8px;border:none;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(320, 240)
        lay.addWidget(self.video_label)

    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def show_no_signal(self):
        self.video_label.setText("NO SIGNAL")
        self.video_label.setStyleSheet(
            "background:#000;border-radius:8px;border:none;"
            "color:%s;font-size:18px;" % TEXT_GRAY
        )


class TimerPanel(RoundedPanel):
    def __init__(self, parent=None):
        super().__init__(BORDER_CYAN, parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(18,10,18,10); lay.setSpacing(10)

        ico = QLabel(ICO_CLOCK)
        ico.setFont(QFont(FONT_EMOJI, 16))
        ico.setStyleSheet("color:%s;background:transparent;border:none;" % TEXT_CYAN)

        ttl = QLabel("\uc2dc\uac04")   # 시간
        ttl.setFont(QFont(FONT_SANS, 18, QFont.Bold))
        ttl.setStyleSheet("color:%s;background:transparent;border:none;" % TEXT_CYAN)

        self.time_label = QLabel("00:00")
        self.time_label.setFont(QFont(FONT_SANS, 22, QFont.Bold))
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.time_label.setStyleSheet("color:%s;background:transparent;border:none;" % TEXT_CYAN)

        lay.addWidget(ico); lay.addWidget(ttl); lay.addStretch(); lay.addWidget(self.time_label)

    def set_time(self, seconds):
        m, s = divmod(seconds, 60)
        self.time_label.setText("%02d:%02d" % (m, s))


class ScorePanel(RoundedPanel):
    def __init__(self, parent=None):
        super().__init__(BORDER_YELLOW, parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20,14,20,14); lay.setSpacing(4)

        hdr = QWidget()
        hdr.setStyleSheet("background:transparent;border:none;")
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(0,0,0,0); hl.setSpacing(8)
        star = QLabel(ICO_STAR)
        star.setFont(QFont(FONT_SANS, 18))
        star.setStyleSheet("color:%s;background:transparent;border:none;" % TEXT_YELLOW)
        ttl = QLabel("\uc810\uc218")   # 점수
        ttl.setFont(QFont(FONT_SANS, 18, QFont.Bold))
        ttl.setStyleSheet("color:%s;background:transparent;border:none;" % TEXT_YELLOW)
        hl.addWidget(star); hl.addWidget(ttl); hl.addStretch()
        lay.addWidget(hdr)

        self.score_label = QLabel("00")
        self.score_label.setAlignment(Qt.AlignCenter)
        self.score_label.setFont(QFont(FONT_SANS, 120, QFont.Bold))
        self.score_label.setStyleSheet("color:%s;background:transparent;border:none;" % TEXT_YELLOW)
        self.score_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self.score_label)

    def set_score(self, score):
        self.score_label.setText("%02d" % score)


class StatusPanel(RoundedPanel):
    start_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(BORDER_CYAN, parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(18,10,18,10); lay.setSpacing(12)

        self.status_icon = QLabel(ICO_WAIT)
        self.status_icon.setFont(QFont(FONT_EMOJI, 18))
        self.status_icon.setStyleSheet("color:%s;background:transparent;border:none;" % TEXT_GRAY)

        self.status_label = QLabel("\ub300\uae30")   # 대기
        self.status_label.setFont(QFont(FONT_SANS, 18, QFont.Bold))
        self.status_label.setStyleSheet("color:%s;background:transparent;border:none;" % TEXT_GRAY)

        lay.addWidget(self.status_icon); lay.addWidget(self.status_label); lay.addStretch()

        self.start_btn = QPushButton("START")
        self.start_btn.setFont(QFont(FONT_SANS, 16, QFont.Bold))
        self.start_btn.setFixedHeight(46); self.start_btn.setMinimumWidth(140)
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self._style_start()
        self.start_btn.clicked.connect(self.start_clicked)
        lay.addWidget(self.start_btn)

    def _style_start(self):
        self.start_btn.setStyleSheet(
            "QPushButton{background:%s;color:%s;border:2px solid %s;"
            "border-radius:10px;padding:4px 24px;letter-spacing:2px;}"
            "QPushButton:hover{background:%s;}"
            % (BTN_BG, TEXT_CYAN, BORDER_CYAN, BTN_HOVER)
        )

    def _style_stop(self):
        self.start_btn.setStyleSheet(
            "QPushButton{background:#5c0d0d;color:#ff6060;border:2px solid #ff4040;"
            "border-radius:10px;padding:4px 24px;letter-spacing:2px;}"
            "QPushButton:hover{background:#7a1010;}"
        )

    def set_status(self, text, icon=None, color=TEXT_GRAY):
        if icon:
            self.status_icon.setText(icon)
        self.status_label.setText(text)
        self.status_icon.setStyleSheet("color:%s;background:transparent;border:none;" % color)
        self.status_label.setStyleSheet(
            "color:%s;background:transparent;border:none;font-size:18px;font-weight:bold;" % color
        )

    def set_running(self, running):
        if running:
            self.start_btn.setText("STOP"); self._style_stop()
        else:
            self.start_btn.setText("START"); self._style_start()


class MainWindow(QMainWindow):
    GAME_DURATION = 60

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FreeKick Bot")
        self.setMinimumSize(1280, 720)
        self.setStyleSheet("QMainWindow{background:%s;}" % BG_COLOR)
        self.score = 0; self.elapsed = 0; self.running = False
        self._build_ui()
        self._setup_camera()
        self._setup_timers()

    def _build_ui(self):
        c = QWidget(); c.setStyleSheet("background:%s;" % BG_COLOR)
        self.setCentralWidget(c)
        root = QHBoxLayout(c)
        root.setContentsMargins(20,20,20,20); root.setSpacing(20)

        self.camera_panel = CameraPanel()
        root.addWidget(self.camera_panel, stretch=5)

        right = QVBoxLayout(); right.setSpacing(14)
        self.timer_panel  = TimerPanel()
        self.score_panel  = ScorePanel()
        self.status_panel = StatusPanel()
        self.timer_panel.setFixedHeight(70)
        self.status_panel.setFixedHeight(70)
        right.addWidget(self.timer_panel)
        right.addWidget(self.score_panel, stretch=1)
        right.addWidget(self.status_panel)
        root.addLayout(right, stretch=5)

        self.status_panel.start_clicked.connect(self._on_start_stop)

    def _setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.camera_panel.show_no_signal(); self.cap = None

    def _setup_timers(self):
        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self._update_frame)
        self.frame_timer.start(33)
        self.game_timer = QTimer(self)
        self.game_timer.timeout.connect(self._tick)

    def _update_frame(self):
        if self.cap is None: return
        ret, frame = self.cap.read()
        if ret: self.camera_panel.update_frame(frame)

    def _tick(self):
        self.elapsed += 1
        self.timer_panel.set_time(self.elapsed)
        if self.elapsed >= self.GAME_DURATION:
            self._end_game()

    def _on_start_stop(self):
        if not self.running: self._start_game()
        else: self._end_game()

    def _start_game(self):
        self.running = True; self.elapsed = 0; self.score = 0
        self.timer_panel.set_time(0); self.score_panel.set_score(0)
        self.status_panel.set_status("\uc9c4\ud589 \uc911", "\u25b6", TEXT_CYAN)  # 진행 중
        self.status_panel.set_running(True)
        self.game_timer.start(1000)

    def _end_game(self):
        self.running = False; self.game_timer.stop()
        self.status_panel.set_status("\uc885\ub8cc", ICO_FLAG, TEXT_YELLOW)  # 종료
        self.status_panel.set_running(False)

    def add_score(self):
        if self.running:
            self.score += 1
            self.score_panel.set_score(self.score)

    def closeEvent(self, event):
        self.frame_timer.stop(); self.game_timer.stop()
        if self.cap is not None: self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
