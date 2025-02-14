import cv2
import numpy as np
import mediapipe as mp
import json
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

def load_mediapipe_models():
    mpFaceMesh = mp.solutions.face_mesh
    face_mesh = mpFaceMesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=2)

    return face_mesh

def get_landmarks(mp_face_mesh_model, image) -> list:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh_model.process(image_rgb)
    
    if not results.multi_face_landmarks:
        print("No face detected.")
        return []
    
    landmarks = results.multi_face_landmarks[0].landmark
    return landmarks
    
def align_face(image, debug = False):
    face_mesh = load_mediapipe_models()
    
    h, w, _ = image.shape
    
    landmarks = get_landmarks(face_mesh, image)
    
    left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])  # Right eye (from image perspective)
    right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])  # Left eye
    
    # Compute the angle
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # Compute center for rotation
    eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    
    # Rotation matrix
    M = cv2.getRotationMatrix2D(eye_center, angle, 1)
    aligned_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
    
    if debug:
        left_eye_new = np.dot(M[:, :2], left_eye) + M[:, 2]
        right_eye_new = np.dot(M[:, :2], right_eye) + M[:, 2]
        cv2.line(aligned_image, tuple(left_eye_new.astype(int)), tuple(right_eye_new.astype(int)), (0, 0, 255), 2)
    
    return aligned_image

def capture_image():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        
        cv2.imshow("Webcam - Press any key to capture", frame)
        
        if cv2.waitKey(1) & 0xFF != 255:
            cv2.imwrite("captured_frame.jpg", frame)
            print("Image captured and saved as 'captured_frame.jpg'")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return frame

def extract_embedding(image):
    # Initialize InsightFace
    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_thresh=0.4)
    
    # Detect and extract features
    faces = app.get(image)
    if not faces:
        return None  # No face detected
    
    # Assuming the first detected face is the one needed
    embedding = faces[0].embedding.tolist()
    
    return embedding

def save_embedding(person_name, embedding, file_path="embeddings.json"):
    data = load_stored_embeddings(file_path)
    
    # Ensure the format is consistent
    if not isinstance(data, dict):
        data = {}  # Reset to an empty dictionary if the structure is incorrect
    
    data[person_name] = embedding  # Assign the new embedding
    
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def load_stored_embeddings(file_path="embeddings.json"):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print("Embedding file not found.")
        return {}

def compare_embedding(input_embedding, file_path="embeddings.json"):
    stored_data = load_stored_embeddings(file_path)
    
    if not stored_data:
        print("No stored embeddings to compare.")
        return None, None
    
    input_embedding = np.array(input_embedding).reshape(1, -1)
    best_match = None
    highest_similarity = -1  # Cosine similarity ranges from -1 to 1
    
    for person_name, stored_embedding in stored_data.items():
        stored_embedding = np.array(stored_embedding).reshape(1, -1)
        similarity = cosine_similarity(input_embedding, stored_embedding)[0][0]
        
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = person_name
    
    return best_match, highest_similarity


if __name__ == "__main__":
    image = capture_image()
    out_image = align_face(image)

    cv2.imshow("Aligned_face", out_image)
    cv2.imwrite("aligned_face.jpg", out_image)
    cv2.waitKey()
    
    embeding = extract_embedding(out_image)
    