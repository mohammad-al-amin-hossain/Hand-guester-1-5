import numpy as np
import cv2
import math

capture = cv2.VideoCapture(0)

while capture.isOpened():


    ret, frame = capture.read()

    cv2.rectangle(frame, (50, 70), (350, 350), (0, 0, 0), 0)
    crop_image = frame[50:340, 70:340]

    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    #binary image with where white will be skin colors
    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))
    kernel = np.ones((5, 5))
    #filter out the background noise
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)

    cv2.imshow("Threshold_win", thresh)
    # Find contours
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    try:
        #find contour with maximum area
        contour = max(contours, key=lambda x: cv2.contourArea(x))
        # bounded rectangle around the contour
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)
        #for convexHall
        hull = cv2.convexHull(contour)
        # draw contour
        drawing = np.zeros(crop_image.shape, np.uint8)
        cv2.drawContours(drawing, [contour], -1, (255, 255, 0), 0)
        cv2.drawContours(drawing, [hull], -1, (139, 69, 19), 0)
        #find convexity defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        count_defects = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

            # draw a circle at the far point
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_image, far, 1, [0, 255, 0], -1)

            cv2.line(crop_image, start, end, [75, 0, 130], 2)


        if count_defects == 0:
            cv2.putText(frame, "ONE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 100, 0),2)
        elif count_defects == 1:
            cv2.putText(frame, "TWO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 100, 0), 2)
        elif count_defects == 2:
            cv2.putText(frame, "THREE", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 100, 0), 2)
        elif count_defects == 3:
            cv2.putText(frame, "FOUR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 100, 0), 2)
        elif count_defects == 4:
            cv2.putText(frame, "FIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 100, 0), 2)
            # cap = cv2.VideoCapture("killing_me_softly.mp4")
            # while True:
            #     ret, frame = cap.read()
            #
            #     cv2.imshow("d", frame)
            #
            #     if cv2.waitKey(0) == ord('b'):
            #         break
            # cap.release()
            # cv2.destroyAllWindows()
        else:
            pass
    except:
        pass

    cv2.imshow("Gesture", frame)
    all_image = np.hstack((drawing, crop_image))
    cv2.imshow('Contours', all_image)


    if cv2.waitKey(1) == ord('a'):
        break


capture.release()
cv2.destroyAllWindows()

capture = cv2.VideoCapture(0)

while capture.isOpened():


    ret, frame = capture.read()

    cv2.rectangle(frame, (50, 70), (350, 350), (0, 0, 0), 0)
    crop_image = frame[50:340, 70:340]

    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    #binary image with where white will be skin colors
    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))
    kernel = np.ones((5, 5))
    #filter out the background noise
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)

    cv2.imshow("Threshold_win", thresh)
    # Find contours
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    try:
        #find contour with maximum area
        contour = max(contours, key=lambda x: cv2.contourArea(x))
        # bounded rectangle around the contour
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)
        #for convexHall
        hull = cv2.convexHull(contour)
        # draw contour
        drawing = np.zeros(crop_image.shape, np.uint8)
        cv2.drawContours(drawing, [contour], -1, (255, 255, 0), 0)
        cv2.drawContours(drawing, [hull], -1, (139, 69, 19), 0)
        #find convexity defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        count_defects = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

            # draw a circle at the far point
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_image, far, 1, [0, 0, 0], -1)

            cv2.line(crop_image, start, end, [75, 0, 130], 2)


        if count_defects == 0:
            cv2.putText(frame, "ONE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 100, 0),2)
        elif count_defects == 1:
            cv2.putText(frame, "TWO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 100, 0), 2)
        elif count_defects == 2:
            cv2.putText(frame, "THREE", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 100, 0), 2)
            # imf = cv2.imread('cat.jpg', 1)
            # cv2.imshow('image', imf)
            # if cv2.waitKey(0) == ord('b'):
            #    break
            # cv2.destroyAllWindows()

        elif count_defects == 3:
            cv2.putText(frame, "FOUR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 100, 0), 2)
        elif count_defects == 4:
            cv2.putText(frame, "FIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 100, 0), 2)
            cap = cv2.VideoCapture("killing_me_softly.mp4")
            while True:
                ret, frame = cap.read()

                cv2.imshow("d", frame)

                if cv2.waitKey(0) == ord('b'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            pass
    except:
        pass

    cv2.imshow("Gesture", frame)
    all_image = np.hstack((drawing, crop_image))
    cv2.imshow('Contours', all_image)


    if cv2.waitKey(1) == ord('a'):
        break


capture.release()
cv2.destroyAllWindows()