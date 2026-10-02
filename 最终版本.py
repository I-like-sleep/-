import cv2
import numpy as np

# 标定参数：橡皮在 9 cm 处，视差约为 15.5 px
K = 140.0

LEFT_INDEX = 0
RIGHT_INDEX = 2
LOWER_HSV = np.array([5, 0, 60], dtype=np.uint8)
UPPER_HSV = np.array([50, 90, 255], dtype=np.uint8)
MORPH_KERNEL = np.ones((9, 9), np.uint8)

capL = cv2.VideoCapture(LEFT_INDEX, cv2.CAP_DSHOW)
capR = cv2.VideoCapture(RIGHT_INDEX, cv2.CAP_DSHOW)


def find_center(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 100:
        return None

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None

    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    return center_x, center_y, contour


try:
    if not capL.isOpened() or not capR.isOpened():
        raise RuntimeError("摄像头打开失败")

    for _ in range(12):
        capL.read()
        capR.read()

    while True:
        okL, frameL = capL.read()
        okR, frameR = capR.read()
        if not okL or not okR:
            break

        frameR = cv2.rotate(frameR, cv2.ROTATE_180)
        left = find_center(frameL)
        right = find_center(frameR)

        if left is not None and right is not None:
            cxL, cyL, contourL = left
            cxR, cyR, contourR = right
            cxR_fixed = frameR.shape[1] - cxR
            disparity = abs(cxL - cxR_fixed)

            if disparity == 0:
                distance = 0.0
            else:
                distance = K / disparity

            cv2.circle(frameL, (cxL, cyL), 5, (0, 255, 0), -1)
            cv2.drawContours(frameL, [contourL], -1, (0, 0, 255), 2)
            cv2.circle(frameR, (cxR, cyR), 5, (0, 255, 0), -1)
            cv2.drawContours(frameR, [contourR], -1, (0, 0, 255), 2)

            cv2.putText(frameL, f"d = {disparity} px", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if disparity != 0:
                cv2.putText(frameL, f"Z = {distance:.1f} cm", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            print(f"L:({cxL},{cyL})  R:({cxR_fixed},{cyR})  d = {disparity} px  Z = {distance:.1f} cm")

        cv2.imshow("Left Camera", frameL)
        cv2.imshow("Right Camera", frameR)

        if cv2.waitKey(1) == 27:
            break
finally:
    for cap in (capL, capR):
        cap.release()
    cv2.destroyAllWindows()