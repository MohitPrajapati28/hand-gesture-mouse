import cv2
import mediapipe as mp
import pyautogui
import math

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

prev_x, prev_y = 0, 0
dragging = False

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:

            x1, y1 = 0, 0  # index
            x2, y2 = 0, 0  # thumb
            x3, y3 = 0, 0  # middle

            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                x, y = int(lm.x * w), int(lm.y * h)

                if id == 8:  # index
                    x1, y1 = x, y

                    # smooth movement
                    screen_x = screen_width * lm.x
                    screen_y = screen_height * lm.y

                    curr_x = prev_x + (screen_x - prev_x) / 3
                    curr_y = prev_y + (screen_y - prev_y) / 3

                    pyautogui.moveTo(curr_x, curr_y)

                    prev_x, prev_y = curr_x, curr_y

                    cv2.circle(img, (x, y), 10, (255, 0, 255), cv2.FILLED)

                if id == 4:  # thumb
                    x2, y2 = x, y

                if id == 12:  # middle finger
                    x3, y3 = x, y

            # distances
            dist_thumb_index = math.hypot(x2 - x1, y2 - y1)
            dist_thumb_middle = math.hypot(x2 - x3, y2 - y3)

            # LEFT CLICK + DRAG
            if dist_thumb_index < 40:
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

            # RIGHT CLICK
            if dist_thumb_middle < 40:
                pyautogui.rightClick()

            # SCROLL
            if y1 < y3:  # index above middle
                pyautogui.scroll(20)
            elif y1 > y3:
                pyautogui.scroll(-20)

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gesture Mouse", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break