from PIL import Image
import numpy as np

def dimensionality_reduction(image_path, threshold=128):

    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)

    img_gray = np.dot(arr[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)

    img_pb = (img_gray > threshold) * 255
    img_pb = img_pb.astype(np.uint8)

    gray_pil = Image.fromarray(img_gray, mode="L")
    pb_pil = Image.fromarray(img_pb, mode="L")

    return gray_pil, pb_pil


if __name__ == "__main__":
    gray, pb = dimensionality_reduction("image.jpeg", limiar=127)

    gray.show(title="Gray Scale")
    pb.show(title="Black and White")

    gray.save("grayscale.png")
    pb.save("blackwhite.png")