import cv2
import os
import pickle
import numpy as np
from datetime import datetime
import time

def create_elegant_overlay(frame, text="", color=(255, 255, 255), progress=0, photo_count=0):
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
    
    title = "FACE REGISTRATION SYSTEM"
    title_size = cv2.getTextSize(title, font, 1, 1)[0]
    title_x = int((width - title_size[0]) / 2)
    
    line_length = 50
    cv2.line(overlay, (title_x - line_length - 10, 35), (title_x - 10, 35), (255, 255, 255), 1)
    cv2.line(overlay, (title_x + title_size[0] + 10, 35), (title_x + title_size[0] + line_length + 10, 35), (255, 255, 255), 1)
    
    cv2.putText(overlay, title, (title_x, 40), font, 1, (255, 255, 255), 1)
    cv2.putText(overlay, current_time, (width-140, 40), font, 0.8, (200, 200, 200), 1)
    cv2.putText(overlay, current_date, (20, 40), font, 0.8, (200, 200, 200), 1)
    
    photo_text = f"Photos: {photo_count}"
    cv2.putText(overlay, photo_text, (width-140, height-80), font, 0.8, (200, 200, 200), 1)
    
    if text:
        text_size = cv2.getTextSize(text, font, 0.9, 1)[0]
        text_x = int((width - text_size[0]) / 2)
        cv2.putText(overlay, text, (text_x+1, height-70+1), font, 0.9, (0, 0, 0), 2)
        cv2.putText(overlay, text, (text_x, height-70), font, 0.9, color, 2)
        
        if progress > 0:
            bar_width = int(width * 0.6)
            bar_height = 8
            bar_x = int((width - bar_width) / 2)
            bar_y = height - 45
            
            cv2.rectangle(overlay, (bar_x-2, bar_y-2), (bar_x + bar_width+2, bar_y + bar_height+2), 
                         (50, 50, 50), -1)
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (30, 30, 30), -1)
            
            progress_width = int(bar_width * progress)
            if progress_width > 0:
                for i in range(progress_width):
                    x = bar_x + i
                    progress_color = list(color)
                    alpha = 0.7 + (i / progress_width) * 0.3
                    progress_color = [int(c * alpha) for c in progress_color]
                    cv2.line(overlay, (x, bar_y), (x, bar_y + bar_height), progress_color, 1)
    
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
    cv2.line(frame, (x, y), (x+l, y), color, t)
    cv2.line(frame, (x, y), (x, y+l), color, t)
    cv2.line(frame, (x+w-l, y), (x+w, y), color, t)
    cv2.line(frame, (x+w, y), (x+w, y+l), color, t)
    cv2.line(frame, (x, y+h-l), (x, y+h), color, t)
    cv2.line(frame, (x, y+h), (x+l, y+h), color, t)
    cv2.line(frame, (x+w-l, y+h), (x+w, y+h), color, t)
    cv2.line(frame, (x+w, y+h), (x+w, y+h-l), color, t)

def register_new_mahasiswa():
    if not os.path.exists('mahasiswa_faces'):
        os.makedirs('mahasiswa_faces')
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    mahasiswa_id = input("Masukkan NIK Mahasiswa: ")
    mahasiswa_name = input("Masukkan Nama Mahasiswa: ")
    
    face_images = []
    photo_count = 0
    last_capture_time = 0
    capture_interval = 1.0
    
    window_name = 'Face Registration System'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Mirror frame
        
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        current_time = time.time()
        instruction_text = "Posisikan wajah Anda di dalam kotak dan tekan SPACE"
        instruction_color = (255, 255, 255)
        
        # Apply base overlay
        frame = create_elegant_overlay(frame, instruction_text, instruction_color, 0, photo_count)
        
        for (x, y, w, h) in faces:
            draw_elegant_box(frame, x, y, w, h, (0, 255, 255))
        
        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and current_time - last_capture_time >= capture_interval:
            if len(faces) == 1:
                x, y, w, h = faces[0]
                face_img = frame[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (128, 128))
                face_images.append(face_img)
                photo_count += 1
                last_capture_time = current_time
                
                if photo_count < 3:
                    print(f"Foto {photo_count}/3 berhasil diambil!")
                else:
                    avg_face = np.mean(face_images, axis=0).astype(np.uint8)
                    
                    mahasiswa_data = {
                        'id': mahasiswa_id,
                        'name': mahasiswa_name,
                        'face_img': avg_face,
                        'face_images': face_images
                    }
                    
                    with open(f'mahasiswa_faces/{mahasiswa_id}.pkl', 'wb') as f:
                        pickle.dump(mahasiswa_data, f)
                    
                    print(f"Registrasi {mahasiswa_name} berhasil dengan {photo_count} foto!")
                    break
            else:
                print("Tidak ada wajah terdeteksi atau terdeteksi lebih dari satu wajah")
        
        elif key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    register_new_mahasiswa() 