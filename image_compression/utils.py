import cv2
import numpy

def load_image(file_path: str) -> numpy.ndarray:
    """Loads an image in grayscale format."""
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {file_path}")
    return numpy.array(img/255.0, dtype=numpy.float32)

