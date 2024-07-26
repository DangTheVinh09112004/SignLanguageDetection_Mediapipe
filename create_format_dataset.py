import os
import cv2
import mediapipe as mp
import pickle
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)
    root = "My_Data"
    categories = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L"]
    points = []
    labels = []
    for category in categories:
        data_files = os.path.join(root, category)
        for item in os.listdir(data_files):
            data1 = []
            cx = []
            cy = []
            item_files = os.path.join(data_files, item)
            image = cv2.imread(item_files)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_hand = hands.process(image)
            if image_hand.multi_hand_landmarks:
                for hand_landmarks in image_hand.multi_hand_landmarks:
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
                points.append(data1)
                labels.append(category)
    print(len(points))
    print(len(labels))
    if os.path.exists("file.pickle"):
        os.remove("file.pickle")
    f = open("file.pickle", "wb")
    pickle.dump({"Points": points, "Labels": labels}, f)
    f.close()


