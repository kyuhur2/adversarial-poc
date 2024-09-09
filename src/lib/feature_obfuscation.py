import cv2
import numpy as np
from .base_modifier import BaseModifier

class FeatureObfuscation(BaseModifier):
    def apply(self, input_image_path: str, output_image_path: str) -> None:
        """Applies feature obfuscation to the input image by blacking out the eyes."""
        
        # load the image
        image: np.ndarray = cv2.imread(input_image_path)
        gray_image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        
        # load Haar cascades for face and eye detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # detect faces
        faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # get the region of interest (the face) from the image
            face_roi = image[y:y+h, x:x+w]
            gray_face_roi = gray_image[y:y+h, x:x+w]

            # detect eyes within the face region
            eyes = eye_cascade.detectMultiScale(gray_face_roi)

            for (ex, ey, ew, eh) in eyes:
                # obfuscate each detected eye by blacking it out (or you can blur it instead)
                cv2.rectangle(face_roi, (ex, ey), (ex + ew, ey + eh), (0, 0, 0), -1)  # black out the eyes

        # save the obfuscated image
        cv2.imwrite(output_image_path, image)
