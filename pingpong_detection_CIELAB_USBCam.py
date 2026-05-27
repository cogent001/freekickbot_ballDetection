import cv2
import numpy as np
import time

def nothing(x):
    pass


cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
cv2.createTrackbar('L_min', 'Controls', 0, 255, nothing)
cv2.createTrackbar('L_max', 'Controls', 255, 255, nothing)
cv2.createTrackbar('A_min', 'Controls', 130, 255, nothing)
cv2.createTrackbar('A_max', 'Controls', 200, 255, nothing)
cv2.createTrackbar('B_min', 'Controls', 130, 255, nothing)
cv2.createTrackbar('B_max', 'Controls', 255, 255, nothing)

# 0번 장치는 일반적으로 첫 번째 웹캠 (/dev/video0)
cap = cv2.VideoCapture(0)

#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

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

    lower_lab = np.array([L_min, A_min, B_min])
    upper_lab = np.array([L_max, A_max, B_max])

    mask = cv2.inRange(lab, lower_lab, upper_lab)

    # 모폴로지 연산으로 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 컨투어 검출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:  # 잡음 제거용 최소 면적
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            if radius > 5:
                center = (int(x), int(y))
                cv2.circle(frame, center, int(radius), (0, 255, 255), 2)
                cv2.putText(frame, f"({int(x)}, {int(y)})", (int(x)-40, int(y)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Orange Ball Detection (Lab)", frame)
    #cv2.imshow("Mask", mask)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()