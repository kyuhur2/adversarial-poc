from abc import ABC, abstractmethod

class BaseModifier(ABC):
    """Abstract base class for image modification techniques."""
    
    @abstractmethod
    def apply(self, input_image_path: str, output_image_path: str) -> None:
        """
        Abstract method to apply the modification to the image.

        Args:
            input_image_path (str): Path to the input image.
            output_image_path (str): Path to save the modified image.
        """
        pass
