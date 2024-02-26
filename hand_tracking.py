import cv2
import mediapipe as mp
import time
import math
import numpy as np


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, 
                                        max_num_hands=self.maxHands, 
                                        min_detection_confidence=self.detectionCon, 
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.results = None


    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img


    def find_position(self, img, hand_no=0, draw=True):
        self.lmList = []
        xList = []
        yList = []
        border_box = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(myHand.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            border_box = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.lmList, border_box


    def fingers_up(self):
        fingers = []

        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:  # if x-coord of point 4 is bigger then x-coord of point 3
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


    def find_distance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def create_frame_rate(img, prev_time):
    cur_time = time.time()
    fps = 1 / (cur_time - prev_time)
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    return cur_time

# def main():
#     pTime = 0
#     cap = cv2.VideoCapture(0)
#     detector = HandDetector()
#     while True:
#         _, img = cap.read()
#         img = cv2.flip(img, 180)
#         img = cv2.resize(img, (980, 720))
#         img = detector.find_hands(img)
#         lmList, border_box = detector.find_position(img)

#         pTime = create_frame_rate(img=img, prev_time=pTime)

#         cv2.imshow("Image", img)

#         if cv2.waitKey(1) == ord(' '):
#             break


# if __name__ == "__main__":
#     main()








# def find_position(self, img, hand_no=0, draw=True):
#         self.lmList = []
#         self.bbox = []
#         if self.results.multi_hand_landmarks:
#             for myHand in self.results.multi_hand_landmarks:
#                 xList = []
#                 yList = []
#                 bbox = []
#                 lmList = []
#                 for id, lm in enumerate(myHand.landmark):
#                     h, w, _ = img.shape
#                     cx, cy = int(lm.x * w), int(lm.y * h)
#                     xList.append(cx)
#                     yList.append(cy)
#                     lmList.append([id, cx, cy])
                    
#                     if draw:
#                         cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                
#                 xmin, xmax = min(xList), max(xList)
#                 ymin, ymax = min(yList), max(yList)
#                 bbox = [xmin, ymin, xmax, ymax]

#                 self.lmList.append(lmList)
#                 self.bbox.append(bbox)

#                 if draw:
#                     cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
#                                 (0, 255, 0), 2)

#         return self.lmList, self.bbox