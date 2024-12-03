import cv2
import face_recognition
from deepface import DeepFace
import mediapipe as mp
import os
from tqdm import tqdm
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desabilitar otimizações específicas
tf.get_logger().setLevel('ERROR')

def detect_pose_and_faces(video_path, output_path):
    """
    Detecta poses, rostos e emoções no vídeo.
    Salva os resultados em um arquivo JSON por frame.
    """
    mp_pose = mp.solutions.pose
    mp_face_detection = mp.solutions.face_detection
    pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, static_image_mode=False)
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir = os.path.join(os.path.dirname(output_path), "output")
    os.makedirs(output_dir, exist_ok=True)

    output_video_path = os.path.join(output_dir, os.path.basename(output_path))
    output_json_path = os.path.join(output_dir, "results.json")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_skip = 5
    results_data = []

    try:
        for frame_count in tqdm(range(total_frames), desc="Processando vídeo"):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Detectar rostos com MediaPipe
                    face_results = face_detection.process(rgb_frame)
                    face_locations = []
                    if face_results.detections:
                        for detection in face_results.detections:
                            bbox = detection.location_data.relative_bounding_box
                            x, y, w, h = (
                                int(bbox.xmin * width),
                                int(bbox.ymin * height),
                                int(bbox.width * width),
                                int(bbox.height * height),
                            )
                            face_locations.append((y, x + w, y + h, x))

                    # Analisar emoções
                    emotions = analyze_emotions(frame, face_locations)

                    # Desenhar retângulos e exibir emoções
                    for (top, right, bottom, left), emotion in zip(face_locations, emotions):
                        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                        cv2.putText(frame, emotion, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    # Detectar poses e atividades
                    results = pose.process(rgb_frame)
                    activity = detect_activity(results.pose_landmarks, mp_pose)

                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # Adicionar resultados do frame
                    results_data.append({
                        "frame": frame_count,
                        "num_faces": len(face_locations),
                        "emotions": emotions,
                        "activity": activity
                    })

                    out.write(frame)
                    cv2.imshow('Video', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                except Exception as e:
                    print(f"Erro ao processar frame {frame_count}: {e}")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        with open(output_json_path, "w", encoding="utf-8") as json_file:
            json.dump(results_data, json_file, indent=4, ensure_ascii=False)

        print(f"Resultados salvos em: {output_json_path}")


def analyze_emotions(frame, face_locations):
    """
    Detectar emoções em cada rosto encontrado no frame.
    """
    emotions = []
    for top, right, bottom, left in face_locations:
        try:
            # Garantir que os índices estão dentro dos limites da imagem
            top = max(0, top)
            right = min(frame.shape[1], right)
            bottom = min(frame.shape[0], bottom)
            left = max(0, left)

            # Extrair a região da face
            face_region = frame[top:bottom, left:right]

            # Verificar se a região da face não é vazia
            if face_region.size == 0:
                emotions.append("Indefinido")
                continue

            # Analisar a emoção da face
            result = DeepFace.analyze(face_region, actions=['emotion'], enforce_detection=False)

            # Tratar o retorno quando for uma lista (múltiplas faces)
            if isinstance(result, list):
                # Pegar a emoção dominante da primeira face detectada (ajuste conforme necessário)
                dominant_emotion = result[0].get('dominant_emotion', "Indefinido")
            else:
                # Pegar a emoção dominante quando é um único dicionário
                dominant_emotion = result.get('dominant_emotion', "Indefinido")

            emotions.append(dominant_emotion)
        
        except Exception as e:
            print(f"Erro na análise de emoções: {e}")
            emotions.append("Indefinido")
    
    return emotions


def detect_activity(pose_landmarks, mp_pose):
    """
    Detectar atividades com base em landmarks de pose.
    """
    if pose_landmarks:
        # Coordenadas principais
        left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        left_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]

        # Variáveis para identificar movimento
        wrist_movement = (
            abs(left_wrist.x - right_wrist.x) > 0.1 or abs(left_wrist.y - right_wrist.y) > 0.1
        )
        circular_motion = (
            abs(left_wrist.x - left_elbow.x) > 0.1 or abs(right_wrist.x - right_elbow.x) > 0.1
        )
        leg_movement = (
            abs(left_ankle.y - left_knee.y) > 0.1 or abs(right_ankle.y - right_knee.y) > 0.1
        )

        # Dançando: movimentos circulares e braços e pernas mexendo
        if circular_motion and leg_movement:
            return "Dançando"

        # Acenando: mão direita ou esquerda mexendo de um lado para o outro
        if abs(left_wrist.x - left_elbow.x) > 0.1 or abs(right_wrist.x - right_elbow.x) > 0.1:
            return "Acenando"

    return "Neutro"


# Configurações do vídeo de entrada e saída
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'video_teste.mp4')
output_video_path = os.path.join(script_dir, 'output_video_otimizado_2.mp4')

# Processar o vídeo
detect_pose_and_faces(input_video_path, output_video_path)
