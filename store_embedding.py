from utils import capture_image, align_face, extract_embedding, save_embedding
import cv2

if __name__ == '__main__':
    while True:
        image = capture_image()
        out_image = align_face(image)

        cv2.imshow("Aligned_face", out_image)
        cv2.imwrite("aligned_face.jpg", out_image)
        key = cv2.waitKey(0) & 0xFF  # Wait for a key press
        if key == ord('q') or key == ord('Q'):  # Check if 'Q' is pressed
            break
    cv2.destroyAllWindows()
    
    embeding = extract_embedding(out_image)
    person_name = input("Input the person_name: ") 
    save_embedding(person_name, embeding)