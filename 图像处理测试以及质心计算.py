import cv2
import numpy as np

def find_object(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower = np.array([5, 0, 60], dtype=np.uint8)
    upper = np.array([50, 90, 255], dtype=np.uint8)
    th = cv2.inRange(hsv, lower, upper)
    # 形态学去噪
    kernel = np.ones((9, 9), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    # 找轮廓
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, th
    # 取最大轮廓
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 100:  # 面积太小忽略
        return None, th
    # 计算质心
    M = cv2.moments(c)
    if M['m00'] == 0:
        return None, th
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy, c), th

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    result, th = find_object(frame)
    if result:
        cx, cy, c = result
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        cv2.drawContours(frame, [c], -1, (0, 0, 255), 2)
        cv2.putText(frame, f"({cx},{cy})", (cx+10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    cv2.imshow("Diff", th)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()