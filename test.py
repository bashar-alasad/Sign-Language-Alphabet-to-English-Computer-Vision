import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="Model/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 40
imgSize = 300
labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", 
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", 
    "T", "V", "W", "X", "Y"
]

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        
        aspectRatio = w / h
        if aspectRatio > 1:
            new_w = imgSize
            new_h = int(imgSize / aspectRatio)
        else:
            new_h = imgSize
            new_w = int(imgSize * aspectRatio)
        
        imgCropResized = cv2.resize(imgCrop, (new_w, new_h))
        start_x = (imgSize - new_w) // 2
        start_y = (imgSize - new_h) // 2
        imgWhite[start_y:start_y+new_h, start_x:start_x+new_w] = imgCropResized
        
        # Preprocess the image for the model
        img_array = cv2.resize(imgWhite, (224, 224))  # Adjust size if needed
        img_array = img_array.astype(np.float32) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        index = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Determine which hand is being used
        hand_type = hand['type']  # This will be 'Left' or 'Right'
        
        # Display the hand type in the top-left corner of the image
        cv2.rectangle(img, (10, 10), (200, 60), (255, 255, 255), -1)
        cv2.putText(img, f"{hand_type} Hand", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the gesture prediction at the bottom of the bounding box
        label = f"{labels[index]} {confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img, (x, y+h), (x+label_w, y+h+label_h+10), (255, 255, 255), -1)
        cv2.putText(img, label, (x, y+h+label_h+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw the bounding box
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()