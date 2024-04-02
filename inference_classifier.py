import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import pickle
import time

# Load the pre-trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define labels dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10: 'K', 11: 'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19: 'T', 20: 'U', 21: 'V',22:'W',23:'X',24:'Y',25:'Z',26:' '}

# Define a global variable to store the current word
current_word = ""
detected_words = []

# Function to process hand gestures
def process_gestures(frame):
    global current_word  # Declare global variable to modify it within the function

    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    sign_detected = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        if len(data_aux) == 42:  # 21 (x, y) coordinates for a single hand
            prediction = model.predict([np.asarray(data_aux)])
            sign_detected = labels_dict[int(prediction[0])]
            # Append the detected character to the current word
            current_word += sign_detected

    return frame, current_word

# Streamlit main function
def main():
    st.title("SIGN-TO-TEXT ALPHABET CONVERSION SYSTEM")
    st.markdown("## Interact Now 🤖!")
    image_placeholder = st.empty()

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    start_time = time.time()  # Start time for alphabet detection

    # Streamlit loop
    while True:
        ret, frame = cap.read()
        if ret:
            frame_processed, current_word = process_gestures(frame)
            if current_word:
                cv2.putText(frame_processed, f"Current Word: {current_word}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2, cv2.LINE_AA)
                st.write(current_word)

            image_placeholder.image(frame_processed, channels="BGR", use_column_width=True)

        # Check if two seconds have elapsed and current word is not empty
        if time.time() - start_time >= 2 and current_word:
            # Append the current word to the list of detected words
            detected_words.append(current_word)
            current_word = ""  # Reset current word
            start_time = time.time()  # Reset start time

        # Check for input from user to stop the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop ends, display the final detected text
    st.write("Final Detected Text:", ''.join(detected_words))

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
