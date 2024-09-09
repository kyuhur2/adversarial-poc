import cv2
import numpy as np
from .base_modifier import BaseModifier

class GeometricDistortion(BaseModifier):
    def apply(self, input_image_path: str, output_image_path: str) -> None:
        """Applies geometric distortion to the input image."""
        image: np.ndarray = cv2.imread(input_image_path)
        rows, cols, _ = image.shape
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

        M = cv2.getAffineTransform(pts1, pts2)
        distorted_image: np.ndarray = cv2.warpAffine(image, M, (cols, rows))
        cv2.imwrite(output_image_path, distorted_image)
