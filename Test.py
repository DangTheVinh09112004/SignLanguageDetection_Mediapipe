import torch
import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


class HandGestureClassifier(nn.Module):
    def __init__(self):
        super(HandGestureClassifier, self).__init__()
        self.fc1 = nn.Linear(42, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, len(categories))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def load_model(filepath):
    model = HandGestureClassifier()
    try:
        with open(filepath, "rb") as f:
            state_dict = pickle.load(f)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def preprocess_hand_landmarks(hand_landmarks, H, W):
    data1 = []
    cx = []
    cy = []
    for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y
        cx.append(x)
        cy.append(y)

    for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y
        data1.append(x - min(cx))
        data1.append(y - min(cy))

    if len(data1) != 42:
        print("Expected 42 features but got", len(data1))
        return None

    x1 = int(min(cx) * W)
    y1 = int(min(cy) * H)
    x2 = int(max(cx) * W)
    y2 = int(max(cy) * H)
    return data1, (x1, y1, x2, y2)


if __name__ == '__main__':
    categories = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L"]
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (640, 480))

    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.25)
    draws = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles

    model = load_model("model1.pickle")
    if model is None:
        print("Model could not be loaded. Exiting...")
        exit()

    while True:
        flag, frame = cap.read()
        if not flag:
            print("Failed to capture frame")
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                draws.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                                     styles.get_default_hand_landmarks_style(),
                                     styles.get_default_hand_connections_style())
                data1, bbox = preprocess_hand_landmarks(hand_landmarks, H, W)
                if data1 is not None:
                    data1 = torch.tensor(data1, dtype=torch.float32).unsqueeze(0)
                    prediction = model(data1)
                    _, predicted = torch.max(prediction, 1)
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, str(categories[predicted.item()]), (x1, y1 - 10),
                                cv2.FONT_HERSHEY_COMPLEX, 1.3, (0, 0, 0), 3,
                                cv2.LINE_AA)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        cv2.imshow("Frame", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()




