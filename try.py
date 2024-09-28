import random
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from collections import deque

# Initialize MediaPipe hands and pose
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Open the class file and read the class names
with open("utils/coco.txt", "r") as my_file:
    class_list = my_file.read().strip().split("\n")

# Generate random colors for class list
detection_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in class_list]

# Load a pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt", "v8")

# Video capture from the default camera (usually the webcam)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Parameters for video buffering
fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second
buffer_length = 10  # seconds
buffer_size = int(fps * buffer_length)  # Calculate buffer size
frame_buffer = deque(maxlen=buffer_size)  # Buffer to store frames
last_action = None  # Store last detected action
action_detection_window = []  # Store actions for analysis

# Function to check if two skeletons are in contact
def are_skeletons_in_contact(landmarks1, landmarks2, threshold=0.1):
    for i in range(len(landmarks1.landmark)):
        x1, y1 = landmarks1.landmark[i].x, landmarks1.landmark[i].y
        for j in range(len(landmarks2.landmark)):
            x2, y2 = landmarks2.landmark[j].x, landmarks2.landmark[j].y
            # Calculate distance
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if distance < threshold:  # Check if distance is less than threshold
                return True
    return False

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Check if frame is read correctly
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame for hands and pose
    hand_results = hands.process(rgb_frame)
    pose_results = pose.process(rgb_frame)

    # Draw full skeleton if landmarks are found
    skeletons = []
    if pose_results.pose_landmarks:
        skeletons.append(pose_results.pose_landmarks)
        mp.solutions.drawing_utils.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Process hand landmarks
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Draw hand landmarks
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Gesture detection logic (punching and side punching)
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Simple punching detection: if index and middle fingers are below wrist
            if (index_finger_tip.y > wrist.y) and (middle_finger_tip.y > wrist.y):
                cv2.putText(frame, "Punching", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                last_action = "Punching"

            # Side punch detection: Check horizontal position
            side_punch_threshold = 0.1
            if (abs(index_finger_tip.x - wrist.x) > side_punch_threshold) or (abs(middle_finger_tip.x - wrist.x) > side_punch_threshold):
                if (index_finger_tip.y > wrist.y) and (middle_finger_tip.y > wrist.y):
                    cv2.putText(frame, "Side Punching", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    last_action = "Side Punching"

    # Predict on the current frame using YOLO
    detect_params = model.predict(source=[frame], conf=0.45, save=False)

    if detect_params:
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), detection_colors[int(clsID)], 3)

            # Check for cell phone detection
            if class_list[int(clsID)] == "cell phone":
                cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 3)  # Red rectangle
                cv2.putText(frame, "Alert: Cell phone detected!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display class name and confidence
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, f"{class_list[int(clsID)]} {round(conf, 3) * 100}%", 
                        (int(bb[0]), int(bb[1]) - 10), font, 0.5, (255, 255, 255), 2)

    # Add the current frame to the buffer
    frame_buffer.append(frame)

    # Check for contact between skeletons
    if len(skeletons) == 2:  # Only check if two skeletons are detected
        if are_skeletons_in_contact(skeletons[0], skeletons[1], threshold=0.1):  # Adjust threshold as needed
            cv2.putText(frame, "Alert: Contact Detected!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow("Object Detection", frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord("q"):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
