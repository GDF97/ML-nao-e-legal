import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

DATA_PATH = os.path.join('MP_Data')
actions = np.array(['a', 'b', 'c', 'd',
                    'e', 'f', 'g', 'h',
                    'i', 'j', 'k', 'l',
                    'm', 'n', 'o', 'p', 
                    'q', 'r', 's', 't',
                    'u', 'v', 'w', 'x',
                    'y', 'z', 'Oi', 'Obrigado',
                    'Te Amo', 'Tchau', 'sim', 'não',
                    'Casa'])
no_sequences = 30
sequence_length = 30

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

cap = cv2.VideoCapture(0)

for action in actions:
    for sequence in range(no_sequences):
        for frame_num in range(sequence_length):
            ret, frame = cap.read()

            image, results = mediapipe_detection(frame, holistic)

            draw_landmarks(image, results)

            if frame_num == 0:
                cv2.putText(image, "STARTING COLLECTION", (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
                cv2.putText(image, "Collecting frames for {} Video Number {}".format(action, sequence), (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                cv2.waitKey(2000)
            else:
                cv2.putText(image, "Collecting frames for {} Video Number {}".format(action, sequence), (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            
            keypoints = extract_keypoints(results)
            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            np.save(npy_path, keypoints)
            cv2.imshow('OpenCV Feed', image)
            
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
