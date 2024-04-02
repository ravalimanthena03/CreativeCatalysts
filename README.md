# CreativeCatalysts
This project aims to transform communication accessibility for the community of deaf and hard-of-hearing people. By utilizing Machine Learning, this breaks down boundaries and promotes inclusive communication like never before by translating sign language motions into text with ease.


//DONE://
ASL Gesture Recognition: Recognizes ASL gestures accurately and efficiently.
Translation to English: Converts ASL gestures into English text with high precision.
User Interface and Accessibility: Offers a user-friendly interface with accessibility features to cater to diverse user needs.

//TO BE DONE://
Multi-language Translation: Utilizes translation APIs to render English text into various languages, ensuring global accessibility.(using datasets and hugging face platform)

//Methodology://
Data Collection and Preprocessing: Created and processed ASL gesture datasets to train the model.

Model Training: Utilized randomforest classifier for training the ML model on ASL gesture recognition and translation tasks.

Evaluation and Optimization: Evaluated the model's performance and optimized it for enhanced accuracy and efficiency.

User Interface and Accessibility: Designed an intuitive user interface with accessibility features for seamless interaction using streamlit.

//Description of each file://

Collect_images.py:
This script captures images from a webcam for each class of hand gestures (A-Z and space).
It saves the images in a directory structure suitable for training a machine learning model.
It uses OpenCV (cv2) for webcam access and image processing.

create_dataset.py:
This script processes the collected images to extract hand landmarks using the MediaPipe Hands library.
It constructs a dataset containing hand landmarks for each image.
It saves the dataset and corresponding labels using Python's pickle module.

train_classifier.py:
This script loads the dataset created in the previous step.
It trains a RandomForestClassifier model from scikit-learn on the dataset.
It evaluates the model's accuracy and saves the trained model using pickle.

inference_classifier.py:
This script loads the trained model and MediaPipe Hands for real-time hand gesture inference.
It defines a function to process hand gestures captured through the webcam.
It uses Streamlit to create a simple web application for real-time inference and display the recognized text.
When executed together, they enable the creation and deployment of a sign language translation system. Additionally, the deployment script (inference_classifier.py) utilizes Streamlit for creating a user-friendly interface.


