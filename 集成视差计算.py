import cv2
import numpy as np

LEFT_INDEX = 0
RIGHT_INDEX = 2
LOWER_HSV = np.array([5, 0, 60], dtype=np.uint8)
UPPER_HSV = np.array([50, 90, 255], dtype=np.uint8)
MORPH_KERNEL = np.ones((9, 9), np.uint8)

capL = cv2.VideoCapture(LEFT_INDEX, cv2.CAP_DSHOW)
capR = cv2.VideoCapture(RIGHT_INDEX, cv2.CAP_DSHOW)


def find_center(frame):
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour = max(contours, key=cv2.contourArea, default=None)
    if contour is None or cv2.contourArea(contour) < 100:
        return None

    moments = cv2.moments(contour)
    if not moments["m00"]:
        return None
    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]), contour


try:
    if not all(cap.isOpened() for cap in (capL, capR)):
        raise RuntimeError("摄像头打开失败")

    for _ in range(12):
        for cap in (capL, capR):
            cap.read()

    while True:
        okL, frameL = capL.read()
        okR, frameR = capR.read()
        if not okL or not okR:
            break

        frameR = cv2.rotate(frameR, cv2.ROTATE_180)
        left = find_center(frameL)
        right = find_center(frameR)

        if left is not None and right is not None:
            cxL, cyL, contL = left
            cxR, cyR, contR = right
            cxR_fixed = frameR.shape[1] - cxR
            d = abs(cxL - cxR_fixed)

            for frame, center, contour in (
                (frameL, (cxL, cyL), contL),
                (frameR, (cxR, cyR), contR),
            ):
                cv2.circle(frame, center, 5, (0, 255, 0), -1)
                cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)

            cv2.putText(frameL, f"d = {d} px", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print(f"L:({cxL},{cyL})  R:({cxR_fixed},{cyR})  d = {d} px")

        cv2.imshow("Left Camera", frameL)
        cv2.imshow("Right Camera", frameR)

        if cv2.waitKey(1) == 27:
            break
finally:
    for cap in (capL, capR):
        cap.release()
    cv2.destroyAllWindows()