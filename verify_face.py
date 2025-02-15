import cv2
import os
import pickle
import numpy as np
from datetime import datetime
import time

def cosine_similarity(A, B):
    return np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))

def create_elegant_overlay(frame, text="", color=(255, 255, 255), confidence=0):
    """Create elegant overlay with text and visual elements"""
    height, width = frame.shape[:2]
    overlay = frame.copy()
    
    top_height = 70
    for i in range(top_height):
        alpha = 1 - (i / top_height) * 0.3
        cv2.rectangle(overlay, (0, i), (width, i+1), (0, 0, 0), -1)
        overlay[i, :] = cv2.addWeighted(overlay[i, :], alpha, frame[i, :], 1-alpha, 0)
    
    bottom_height = 120
    for i in range(bottom_height):
        y = height - bottom_height + i
        alpha = 0.7 + (i / bottom_height) * 0.3
        cv2.rectangle(overlay, (0, y), (width, y+1), (0, 0, 0), -1)
        overlay[y, :] = cv2.addWeighted(overlay[y, :], alpha, frame[y, :], 1-alpha, 0)
    
    cv2.line(overlay, (0, top_height), (width, top_height), (200, 200, 200), 1)
    cv2.line(overlay, (0, height-bottom_height), (width, height-bottom_height), (200, 200, 200), 1)
    
    current_time = datetime.now().strftime("%H:%M:%S")
    current_date = datetime.now().strftime("%d %B %Y")
    font = cv2.FONT_HERSHEY_DUPLEX
    
    title = "FACE VERIFICATION SYSTEM"
    title_size = cv2.getTextSize(title, font, 1, 1)[0]
    title_x = int((width - title_size[0]) / 2)
    
    line_length = 50
    cv2.line(overlay, (title_x - line_length - 10, 35), (title_x - 10, 35), (255, 255, 255), 1)
    cv2.line(overlay, (title_x + title_size[0] + 10, 35), (title_x + title_size[0] + line_length + 10, 35), (255, 255, 255), 1)
    
    cv2.putText(overlay, title, (title_x, 40), font, 1, (255, 255, 255), 1)
    cv2.putText(overlay, current_time, (width-140, 40), font, 0.8, (200, 200, 200), 1)
    cv2.putText(overlay, current_date, (20, 40), font, 0.8, (200, 200, 200), 1)
    
    if text:
        text_size = cv2.getTextSize(text, font, 0.9, 1)[0]
        text_x = int((width - text_size[0]) / 2)
        cv2.putText(overlay, text, (text_x+1, height-70+1), font, 0.9, (0, 0, 0), 2)
        cv2.putText(overlay, text, (text_x, height-70), font, 0.9, color, 2)
        
        bar_width = int(width * 0.6)
        bar_height = 8
        bar_x = int((width - bar_width) / 2)
        bar_y = height - 45
        
        cv2.rectangle(overlay, (bar_x-2, bar_y-2), (bar_x + bar_width+2, bar_y + bar_height+2), 
                     (50, 50, 50), -1)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (30, 30, 30), -1)
        
        progress_width = int(bar_width * confidence)
        if progress_width > 0:
            for i in range(progress_width):
                x = bar_x + i
                progress_color = list(color)
                alpha = 0.7 + (i / progress_width) * 0.3
                progress_color = [int(c * alpha) for c in progress_color]
                cv2.line(overlay, (x, bar_y), (x, bar_y + bar_height), progress_color, 1)
        
        conf_text = f"Confidence Level: {int(confidence*100)}%"
        conf_size = cv2.getTextSize(conf_text, font, 0.6, 1)[0]
        conf_x = bar_x + bar_width + 15
        conf_y = bar_y + 6
        
        cv2.putText(overlay, conf_text, (conf_x+1, conf_y+1), font, 0.6, (0, 0, 0), 1)
        cv2.putText(overlay, conf_text, (conf_x, conf_y), font, 0.6, (200, 200, 200), 1)
    
    return overlay

def draw_elegant_box(frame, x, y, w, h, color):
    """Draw elegant face detection box with enhanced visual effects"""
    l = 40
    t = 2
    
    glow_size = 3
    for i in range(glow_size):
        alpha = 1 - (i / glow_size)
        glow_color = tuple(int(c * alpha) for c in color)
        
        # Top left
        cv2.line(frame, (x-i, y-i), (x+l, y-i), glow_color, 1)
        cv2.line(frame, (x-i, y-i), (x-i, y+l), glow_color, 1)
        
        # Top right
        cv2.line(frame, (x+w-l, y-i), (x+w+i, y-i), glow_color, 1)
        cv2.line(frame, (x+w+i, y-i), (x+w+i, y+l), glow_color, 1)
        
        # Bottom left
        cv2.line(frame, (x-i, y+h-l), (x-i, y+h+i), glow_color, 1)
        cv2.line(frame, (x-i, y+h+i), (x+l, y+h+i), glow_color, 1)
        
        # Bottom right
        cv2.line(frame, (x+w-l, y+h+i), (x+w+i, y+h+i), glow_color, 1)
        cv2.line(frame, (x+w+i, y+h-l), (x+w+i, y+h+i), glow_color, 1)
    
    # Draw main corners
    # Top left
    cv2.line(frame, (x, y), (x+l, y), color, t)
    cv2.line(frame, (x, y), (x, y+l), color, t)
    
    # Top right
    cv2.line(frame, (x+w-l, y), (x+w, y), color, t)
    cv2.line(frame, (x+w, y), (x+w, y+l), color, t)
    
    # Bottom left
    cv2.line(frame, (x, y+h-l), (x, y+h), color, t)
    cv2.line(frame, (x, y+h), (x+l, y+h), color, t)
    
    # Bottom right
    cv2.line(frame, (x+w-l, y+h), (x+w, y+h), color, t)
    cv2.line(frame, (x+w, y+h), (x+w, y+h-l), color, t)

def verify_face():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Higher resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    mahasiswa_faces = {}
    for filename in os.listdir('mahasiswa_faces'):
        if filename.endswith('.pkl'):
            with open(os.path.join('mahasiswa_faces', filename), 'rb') as f:
                mahasiswa_data = pickle.load(f)
                mahasiswa_data['face_vector'] = mahasiswa_data['face_img'].flatten().astype(np.float32)
                mahasiswa_faces[mahasiswa_data['id']] = mahasiswa_data
    
    last_verification_time = 0
    verification_interval = 0.3
    
    last_verification_result = None
    last_face_position = None
    
    scale_factor = 1.3
    min_neighbors = 5
    
    window_name = 'Face Verification System'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30)
        )
        
        faces = [(int(x * 2), int(y * 2), int(w * 2), int(h * 2)) for x, y, w, h in faces]
        
        current_time = time.time()
        
        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            last_face_position = (x, y, w, h)
            
            if current_time - last_verification_time >= verification_interval:
                face_img = frame[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (128, 128))
                current_face = face_img.flatten().astype(np.float32)
                
                best_match = None
                highest_similarity = 0
                
                for mah_id, mah_data in mahasiswa_faces.items():
                    similarity = cosine_similarity(
                        current_face,
                        mah_data['face_vector']
                    )
                    
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_match = mah_data
                
                if highest_similarity > 0.85:
                    text = f"Welcome, {best_match['name']} (NIM: {best_match['id']})"
                    color = (0, 255, 0)
                else:
                    text = "Access Denied - Identity Not Verified"
                    color = (0, 0, 255)
                
                last_verification_result = {
                    'text': text,
                    'color': color,
                    'confidence': highest_similarity
                }
                last_verification_time = current_time
        
        frame = create_elegant_overlay(frame)
        
        if last_verification_result and last_face_position:
            x, y, w, h = last_face_position
            draw_elegant_box(frame, x, y, w, h, last_verification_result['color'])
            frame = create_elegant_overlay(
                frame,
                last_verification_result['text'],
                last_verification_result['color'],
                last_verification_result['confidence']
            )
        
        cv2.imshow(window_name, frame)
        
        if len(faces) == 0 and last_verification_time and (current_time - last_verification_time > 2.0):
            last_verification_result = None
            last_face_position = None
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    verify_face() 