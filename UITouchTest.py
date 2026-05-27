import sys
import threading
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal, QPoint

from evdev import InputDevice, list_devices, ecodes


def find_touch_device():
    for path in list_devices():
        dev = InputDevice(path)
        name = dev.name.lower()
        if "touch" in name or "ads7846" in name or "xpt2046" in name:
            return dev
    return None


class TouchTest(QWidget):
    raw_touch_signal = pyqtSignal(int, int, bool)

    def __init__(self, target_screen):
        super().__init__()

        self.target_geo = target_screen.geometry()
        self.count = 0
        self._touch_started_on_button = False

        self.label = QLabel("LCD 터치 테스트")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 28px;")

        self.button = QPushButton("여기를 터치하세요")
        self.button.setStyleSheet("font-size: 26px; padding: 30px;")
        self.button.clicked.connect(self.on_clicked)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)

        self.setLayout(layout)
        self.setWindowTitle("PyQt Touch Test")

        self.raw_touch_signal.connect(self.handle_raw_touch)
        self.start_touch_reader()

    def on_clicked(self):
        self.count += 1
        self.label.setText(f"터치 입력 감지: {self.count}회")

    def handle_raw_touch(self, global_x, global_y, touching):
        global_pos = QPoint(global_x, global_y)
        local_pos = self.button.mapFromGlobal(global_pos)
        inside_button = self.button.rect().contains(local_pos)

        if touching:
            self.button.setDown(inside_button)
            if inside_button:
                self._touch_started_on_button = True
        else:
            should_click = self._touch_started_on_button and inside_button
            self.button.setDown(False)
            self._touch_started_on_button = False
            if should_click:
                self.on_clicked()

    def start_touch_reader(self):
        dev = find_touch_device()
        if dev is None:
            self.label.setText("터치 장치를 찾지 못했습니다.")
            return

        thread = threading.Thread(target=self.read_touch_events, args=(dev,), daemon=True)
        thread.start()

    def read_touch_events(self, dev):
        try:
            xinfo = dev.absinfo(ecodes.ABS_X)
            yinfo = dev.absinfo(ecodes.ABS_Y)
        except Exception:
            return

        xmin, xmax = xinfo.min, xinfo.max
        ymin, ymax = yinfo.min, yinfo.max

        raw_x = None
        raw_y = None
        touching = False

        for event in dev.read_loop():
            if event.type == ecodes.EV_ABS:
                if event.code == ecodes.ABS_X:
                    raw_x = event.value
                elif event.code == ecodes.ABS_Y:
                    raw_y = event.value

            elif event.type == ecodes.EV_KEY and event.code == ecodes.BTN_TOUCH:
                touching = event.value == 1

            elif event.type == ecodes.EV_SYN and raw_x is not None and raw_y is not None:
                sx = (raw_x - xmin) / max(1, xmax - xmin)
                sy = (raw_y - ymin) / max(1, ymax - ymin)

                gx = self.target_geo.x() + int(sx * self.target_geo.width())
                gy = self.target_geo.y() + int(sy * self.target_geo.height())

                gx = max(self.target_geo.x(), min(self.target_geo.x() + self.target_geo.width() - 1, gx))
                gy = max(self.target_geo.y(), min(self.target_geo.y() + self.target_geo.height() - 1, gy))

                self.raw_touch_signal.emit(gx, gy, touching)


app = QApplication(sys.argv)

screens = app.screens()

for i, screen in enumerate(screens):
    print(i, screen.name(), screen.geometry())

# 보통 SPI-1 LCD가 두 번째 화면으로 잡힐 가능성이 큽니다.
# 출력된 screen.name()을 보고 SPI-1에 해당하는 번호를 고르십시오.
target_screen = None

for screen in screens:
    if "SPI" in screen.name():
        target_screen = screen
        break

if target_screen is None:
    target_screen = screens[0]

window = TouchTest(target_screen)

geo = target_screen.geometry()

window.move(geo.x(), geo.y())
window.resize(geo.width(), geo.height())
window.showFullScreen()

sys.exit(app.exec_())
