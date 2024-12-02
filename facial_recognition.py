import cv2
import face_recognition
import mediapipe as mp
import os
from tqdm import tqdm

def detect_pose_and_faces(video_path, output_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, static_image_mode=False)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_skip = 5  # Processar 1 em cada 5 frames
    for frame_count in tqdm(range(total_frames), desc="Processando vídeo"):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            try:
                # Reduzir o tamanho do frame para processamento
                scale_percent = 50  # Reduz para 50% do tamanho original
                small_frame = cv2.resize(frame, (width // 2, height // 2))

                # Converter para RGB
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Detectar rostos
                face_locations = face_recognition.face_locations(rgb_frame, model='hog')

                # Ajustar as coordenadas de volta ao tamanho original
                face_locations = [
                    (top * 2, right * 2, bottom * 2, left * 2)
                    for (top, right, bottom, left) in face_locations
                ]

                # Detectar pose
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Desenhar retângulos ao redor dos rostos
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

                # Desenhar landmarks das poses
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            except Exception as e:
                print(f"Erro ao processar frame: {e}")

        out.write(frame)

        # Mostrar frame (comentar para melhorar desempenho)
        # cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'video_teste.mp4')
output_video_path = os.path.join(script_dir, 'video_processado_4.mp4')

detect_pose_and_faces(input_video_path, output_video_path)
