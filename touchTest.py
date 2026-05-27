import tkinter as tk
import threading
import queue
import sys
from evdev import InputDevice, list_devices, ecodes

# 현재 LCD가 320x480 세로 화면으로 잡혀 있는 경우
W, H = 320, 480

# LCD를 480x320 가로로 회전해서 쓰는 경우에는 위 줄 대신 아래를 사용하십시오.
# W, H = 480, 320

event_queue = queue.Queue()


def find_touch_device():
    for path in list_devices():
        dev = InputDevice(path)
        name = dev.name.lower()

        if "touch" in name or "ads7846" in name or "xpt2046" in name:
            return dev

    return None


def read_touch_events(dev):
    try:
        xinfo = dev.absinfo(ecodes.ABS_X)
        yinfo = dev.absinfo(ecodes.ABS_Y)
    except Exception:
        print("ABS_X / ABS_Y 정보를 읽을 수 없습니다.")
        return

    xmin, xmax = xinfo.min, xinfo.max
    ymin, ymax = yinfo.min, yinfo.max

    x = None
    y = None
    touching = False

    print(f"Touch device: {dev.name}")
    print(f"Input device: {dev.path}")
    print(f"X range: {xmin} ~ {xmax}")
    print(f"Y range: {ymin} ~ {ymax}")

    for event in dev.read_loop():
        if event.type == ecodes.EV_ABS:
            if event.code == ecodes.ABS_X:
                x = event.value
            elif event.code == ecodes.ABS_Y:
                y = event.value

        elif event.type == ecodes.EV_KEY:
            if event.code == ecodes.BTN_TOUCH:
                touching = event.value == 1

        elif event.type == ecodes.EV_SYN:
            if touching and x is not None and y is not None:
                sx = (x - xmin) / max(1, xmax - xmin)
                sy = (y - ymin) / max(1, ymax - ymin)

                px = int(sx * W)
                py = int(sy * H)

                px = max(0, min(W - 1, px))
                py = max(0, min(H - 1, py))

                event_queue.put((px, py, x, y))


def draw_loop():
    try:
        while True:
            px, py, raw_x, raw_y = event_queue.get_nowait()

            canvas.delete("dot")

            r = 10
            canvas.create_oval(
                px - r,
                py - r,
                px + r,
                py + r,
                fill="red",
                tags="dot"
            )

            label.config(
                text=f"screen=({px}, {py}) raw=({raw_x}, {raw_y})"
            )

    except queue.Empty:
        pass

    root.after(20, draw_loop)


dev = find_touch_device()

if dev is None:
    print("터치 장치를 찾지 못했습니다.")
    print("libinput list-devices 명령으로 터치 장치가 보이는지 확인하십시오.")
    sys.exit(1)


root = tk.Tk()
root.title("LCD Touch Test")

# LCD가 wlr-randr에서 0,0 위치에 있다고 가정
root.geometry(f"{W}x{H}+0+0")

# 창 테두리 제거
root.overrideredirect(True)

canvas = tk.Canvas(root, width=W, height=H, bg="black")
canvas.pack(fill="both", expand=True)

label = tk.Label(
    root,
    text="LCD를 터치해 보십시오. 종료: ESC",
    fg="white",
    bg="black",
    font=("Arial", 12)
)
label.place(x=10, y=10)

root.bind("<Escape>", lambda e: root.destroy())

thread = threading.Thread(
    target=read_touch_events,
    args=(dev,),
    daemon=True
)
thread.start()

draw_loop()
root.mainloop()
