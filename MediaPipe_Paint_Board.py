import cv2
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import numpy as np

def draw_ball_location(image, locations):
    for i in range(len(locations)-1):
        if locations[i] is None or locations[i+1] is None:
            continue
        cv2.line(image, tuple(locations[i]), tuple(locations[i+1]), (0, 255, 255), 3)
    return image

list_ball_location = []
history_ball_locations = []

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

text = ""

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

  while cap.isOpened():
    success, image = cap.read()

    if not success:
      print("Ignoring empty camera frame.")
      continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hands.process(image)

    image.flags.writeable = True
    image_height, image_width, _ = image.shape

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:

        thumb_finger_state = 0
        if (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height >
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height >
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height >
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height):
          thumb_finger_state = 1

        index_finger_state = 0
        if (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height >
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height >
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height >
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height):
          index_finger_state = 1

        middle_finger_state = 0
        if (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height >
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height >
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height >
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height):
          middle_finger_state = 1

        ring_finger_state = 0
        if (hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height >
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height >
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height >
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height):
          ring_finger_state = 1

        pinky_finger_state = 0
        if (hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height >
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height >
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height >
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height):
          pinky_finger_state = 1

        use_pen = False
        if (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height >
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height and
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height >
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height and
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height >
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height and
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height >
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height and
            index_finger_state):
          use_pen = True

        use_eraser = False
        if (thumb_finger_state == 1 and index_finger_state == 1 and
            middle_finger_state == 1 and ring_finger_state == 1 and
            pinky_finger_state == 1):
          use_eraser = True

        use_move = False
        if (index_finger_state == 0 and middle_finger_state == 0 and
            ring_finger_state == 0 and pinky_finger_state == 0):
          use_move = True

        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        draw = ImageDraw.Draw(pil_image)

        text = ""
        if use_pen:
          text = "펜"
        elif use_eraser:
          text = "지우개"
        elif use_move:
          text = "이동"

        if text == '펜':
            x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)
            y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
            list_ball_location.append((x, y))

        elif text == '이동':
            history_ball_locations.append(list_ball_location.copy())
            list_ball_location.clear()

        elif text == '지우개':
            history_ball_locations.clear()
            list_ball_location.clear()

        font = ImageFont.truetype("fonts/gulim.ttc", 80)
        bbox = font.getbbox(text)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        x_pos = 50
        y_pos = 50

        draw.rectangle((x_pos, y_pos, x_pos + w, y_pos + h), fill='black')
        draw.text((x_pos, y_pos), text, font=font, fill=(255, 255, 255))

        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    image = draw_ball_location(image, list_ball_location)

    for ball_locations in history_ball_locations:
      image = draw_ball_location(image, ball_locations)

    cv2.imshow('MediaPipe Hands', image)

    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
