import cv2
import numpy as np

def read_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    return image

def resize(image: np.ndarray, new_size: list[int, int], color: list[int, int] = [0, 0, 0]) -> np.ndarray:
    # im = cv2.imread(path)
    old_size = image.shape[:2]
    ratio = min(new_size[0]/ old_size[0], new_size[1]/old_size[1])

    dimensions = (int(old_size[1]*ratio), int(old_size[0]*ratio))
    image = cv2.resize(image, dimensions)
    
    delta_w = new_size[1] - dimensions[0]
    delta_h = new_size[0] - dimensions[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    # color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    
    return new_image


def increase_constrast(image: np.ndarray, limit: int = 2.0, gridsize: tuple[int, ...]= (8, 8)) -> np.ndarray:
    
    # converting to LAB color space
    lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit= limit, tileGridSize= gridsize)
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limage = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_image = cv2.cvtColor(limage, cv2.COLOR_LAB2BGR)

    # Stacking the original image with the enhanced image

    return enhanced_image


def laplance_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    # Declare the variables we are going to use
    ddepth = cv2.CV_16S

    # Convert the image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Laplace function
    dst = cv2.Laplacian(image_gray, ddepth, ksize=kernel_size)

    # converting back to uint8
    abs_dst = cv2.convertScaleAbs(dst)
    result = np.expand_dims(abs_dst, 2)
    # print(result.typ)

    return result


def preprocess_pipeline(image: np.ndarray, input_size: list[int, int], method: str = 'laplance') -> np.ndarray:
    image = resize(image, input_size)
    if method == 'laplance':
        image = laplance_filter(image)
        image = image.repeat(3, axis = -1)
    elif method == 'contrast':
        image = increase_constrast(image)
    else:
        raise Exception('Method must be filled')
    return image