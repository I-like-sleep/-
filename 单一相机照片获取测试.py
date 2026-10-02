import cv2
import numpy as np

# 固定使用两台外接摄像头：
# L = 索引 1
# R = 索引 2
# 内置摄像头(如果存在)忽略不使用
LEFT_INDEX = 1
RIGHT_INDEX = 2

caps = {
    "L": cv2.VideoCapture(LEFT_INDEX, cv2.CAP_DSHOW),
    "R": cv2.VideoCapture(RIGHT_INDEX, cv2.CAP_DSHOW),
}


def find_object(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower = np.array([5, 0, 60], dtype=np.uint8)
    upper = np.array([50, 90, 255], dtype=np.uint8)
    th = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((9, 9), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, th
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 100:
        return None, th
    moments = cv2.moments(c)
    if moments["m00"] == 0:
        return None, th
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return (cx, cy, c), th

try:
    for name, cap in caps.items():
        if not cap.isOpened():
            raise RuntimeError(f"{name} 摄像头打开失败")

    print(f"已固定指定：L = 索引 {LEFT_INDEX}, R = 索引 {RIGHT_INDEX}")

    for cap in caps.values():
        for _ in range(12):
            ret, _ = cap.read()
            if not ret:
                raise RuntimeError("读取摄像头失败")

    for name, cap in caps.items():
        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"{name} 摄像头抓图失败")

        if name == "R":
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        result, _ = find_object(frame)
        if result is None:
            print(f"{name}: 没有找到目标")
        else:
            cx, cy, contour = result
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)
            cv2.putText(frame, f"({cx}, {cy})", (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print(f"{name} 质心: ({cx}, {cy})")

        if not cv2.imwrite(f"{name}.jpg", frame):
            raise RuntimeError(f"{name}.jpg 保存失败")
        print(f"{name}.jpg 已保存，尺寸={frame.shape}")
finally:
    for cap in caps.values():
        cap.release()