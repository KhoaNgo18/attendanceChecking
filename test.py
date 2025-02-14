from utils import capture_image, align_face, extract_embedding, compare_embedding
import cv2

if __name__ == '__main__':
    image = capture_image()
    out_image = align_face(image)

    embeding = extract_embedding(out_image)
    print(compare_embedding(embeding))