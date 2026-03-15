import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# ----- MediaPipe -----
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ----- PyAutoGUI -----
screen_w, screen_h = pyautogui.size()
cursor_x, cursor_y = screen_w // 2, screen_h // 2
last_click = 0
pyautogui.FAILSAFE = False

# ----- Настройки мертвої зони -----
DEAD_ZONE_RADIUS = 40  # пікселів на фронтальній камері
SPEED_FACTOR = 0.3  # множник для повільного руху


# ----- Допоміжні функції -----
def distance(p1, p2, w, h):
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return np.hypot(x2 - x1, y2 - y1)


def process_front(frame, cursor_x, cursor_y):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if res.multi_hand_landmarks:
        for idx, hand in enumerate(res.multi_hand_landmarks):
            label = res.multi_handedness[idx].classification[0].label
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            if label == "Right":
                base = hand.landmark[5]
                index = hand.landmark[8]
                bx, by = int(base.x * w), int(base.y * h)
                ix, iy = int(index.x * w), int(index.y * h)

                # малюємо мертву зону
                cv2.circle(frame, (bx, by), DEAD_ZONE_RADIUS, (255, 0, 0), 2)
                dx, dy = ix - bx, iy - by
                dist = np.hypot(dx, dy)

                if dist > DEAD_ZONE_RADIUS:
                    factor = ((dist - DEAD_ZONE_RADIUS) / 100) * 5  # чим далі, тим швидше
                    factor = max(factor, SPEED_FACTOR)  # мінімальна швидкість
                    cursor_x += dx * factor / 20
                    cursor_y += dy * factor / 20

                cv2.arrowedLine(frame, (bx, by), (ix, iy), (0, 255, 0), 2)
                cv2.circle(frame, (ix, iy), 8, (0, 255, 0), cv2.FILLED)
    return frame, cursor_x, cursor_y


def process_side(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    lkm = False
    pkm = False

    # лінії ЛКМ і ПКМ
    line_y_lkm = int(h * 0.3)
    line_y_pkm = int(h * 0.7)
    cv2.line(frame, (0, line_y_lkm), (w, line_y_lkm), (0, 255, 0), 2)
    cv2.putText(frame, "LKM", (10, line_y_lkm - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.line(frame, (0, line_y_pkm), (w, line_y_pkm), (0, 0, 255), 2)
    cv2.putText(frame, "PKM", (10, line_y_pkm - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if res.multi_hand_landmarks:
        for idx, hand in enumerate(res.multi_hand_landmarks):
            label = res.multi_handedness[idx].classification[0].label
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            if label == "Left":
                index = hand.landmark[8]
                ix = int(index.x * w)
                iy = int(index.y * h)
                # визначення ЛКМ/ПКМ по вертикалі
                if iy < line_y_lkm:
                    lkm = True
                    cv2.circle(frame, (ix, iy), 10, (0, 255, 0), cv2.FILLED)
                elif iy > line_y_pkm:
                    pkm = True
                    cv2.circle(frame, (ix, iy), 10, (0, 0, 255), cv2.FILLED)
                else:
                    # в мертвій зоні
                    cv2.circle(frame, (ix, iy), 10, (255, 0, 0), cv2.FILLED)
    return frame, lkm, pkm


# ----- Відкриття камер -----
cap_front = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap_side = cv2.VideoCapture(1, cv2.CAP_MSMF)
if not cap_front.isOpened():
    print("Фронтальна камера недоступна")
    exit()
if not cap_side.isOpened():
    print("Бокова камера недоступна")

while True:
    # фронтальна камера
    ret_f, frame_f = cap_front.read()
    if ret_f:
        frame_f = cv2.flip(frame_f, 1)
        frame_f, cursor_x, cursor_y = process_front(frame_f, cursor_x, cursor_y)
        cursor_x = max(0, min(screen_w - 1, cursor_x))
        cursor_y = max(0, min(screen_h - 1, cursor_y))
        pyautogui.moveTo(cursor_x, cursor_y)
        cv2.imshow("Front Camera (Cursor Control)", frame_f)

    # бокова камера
    ret_s, frame_s = cap_side.read()
    lkm = pkm = False
    if ret_s:
        frame_s = cv2.flip(frame_s, 1)
        frame_s, lkm, pkm = process_side(frame_s)
        if lkm:
            pyautogui.click()
        if pkm:
            pyautogui.rightClick()
        cv2.imshow("Side Camera (LKM/PKM)", frame_s)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap_front.release()
cap_side.release()
cv2.destroyAllWindows()
