import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

pose = mp_pose.Pose()

screen_w, screen_h = pyautogui.size()

cap = cv2.VideoCapture(0)

prev_x, prev_y = 0, 0
smooth = 5

last_click = 0

def finger_up(hand, tip, pip):
    return hand.landmark[tip].y < hand.landmark[pip].y

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_result = hands.process(rgb)
    pose_result = pose.process(rgb)

    h, w, _ = frame.shape

    # ---------- ОБВЕДЕННЯ ЛЮДИНИ ----------
    if pose_result.pose_landmarks:

        xs = []
        ys = []

        for lm in pose_result.pose_landmarks.landmark:
            xs.append(lm.x)
            ys.append(lm.y)

        x1 = int(min(xs) * w)
        y1 = int(min(ys) * h)
        x2 = int(max(xs) * w)
        y2 = int(max(ys) * h)

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)

    # ---------- РУКИ ----------
    if hand_result.multi_hand_landmarks:

        for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):

            label = hand_result.multi_handedness[idx].classification[0].label

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[8]

            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            # ---------- ПРАВА РУКА (КУРСОР) ----------
            if label == "Right":

                screen_x = np.interp(x,[0,w],[0,screen_w])
                screen_y = np.interp(y,[0,h],[0,screen_h])

                cursor_x = prev_x + (screen_x - prev_x) / smooth
                cursor_y = prev_y + (screen_y - prev_y) / smooth

                pyautogui.moveTo(cursor_x,cursor_y)

                prev_x, prev_y = cursor_x, cursor_y

            # ---------- ЛІВА РУКА (КЛІКИ) ----------
            if label == "Left":

                index = finger_up(hand_landmarks,8,6)
                middle = finger_up(hand_landmarks,12,10)
                ring = finger_up(hand_landmarks,16,14)
                pinky = finger_up(hand_landmarks,20,18)

                thumb = hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x

                # 🔫 пістолет = ЛКМ
                if index and thumb and not middle and not ring and not pinky:
                    if time.time() - last_click > 0.4:
                        pyautogui.click()
                        last_click = time.time()

                # ✌ два пальці = ПКМ
                if index and middle and not ring and not pinky:
                    if time.time() - last_click > 0.4:
                        pyautogui.rightClick()
                        last_click = time.time()

    cv2.imshow("Gesture Control PRO",frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
