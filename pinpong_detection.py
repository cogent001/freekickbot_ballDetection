from picamera2 import Picamera2
import cv2
import numpy as np
import time

# PiCamera2 객체 생성 및 미리보기 해상도 설정
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280,720)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

# 카메라가 안정화되도록 약간 대기
time.sleep(2)

while True:
    frame = picam2.capture_array()

    # 블러로 노이즈 제거
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    # HSV 색공간으로 변환
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 주황색 범위 설정
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # 컨투어 검출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5:
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            if radius > 2:
                center = (int(x), int(y))
                cv2.circle(frame, center, int(radius), (0, 255, 255), 2)
                cv2.putText(frame, f"({int(x)}, {int(y)})", (int(x)-40, int(y)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 출력
    cv2.imshow("Orange Circle Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키
        break

cv2.destroyAllWindows()
picam2.stop()
