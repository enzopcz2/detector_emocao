import cv2
import numpy as np
from tensorflow.keras.models import load_model

# === 1. Carrega o modelo treinado ===
model = load_model("model_emocoes.h5")

emotion_labels = ['Raiva', 'Desgosto', 'Medo', 'Feliz', 'Neutro', 'Triste', 'Surpreso']

# === 2. Inicia webcam ===
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cv2.namedWindow("Detector de Emoções", cv2.WINDOW_NORMAL)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (48, 48)).reshape(1, 48, 48, 1).astype('float32') / 255.0

        prediction = model.predict(roi_resized)
        emotion_idx = np.argmax(prediction)
        confidence = round(np.max(prediction) * 100, 2)
        emotion_text = f"{emotion_labels[emotion_idx]} ({confidence}%)"

        color = (0, 255, 0) if confidence >= 60 else (0, 0, 255)

        cv2.rectangle(gray, (x, y), (x+w, y+h), color, 2)
        cv2.putText(gray, emotion_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Detector de Emoções", gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()