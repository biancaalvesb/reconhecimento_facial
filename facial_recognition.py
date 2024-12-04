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

    frame_skip = 3
    results_data = []
    frames_analyzed = 0
    anomalies_detected = 0

    try:
        for frame_count in tqdm(range(total_frames), desc="Processando vídeo"):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                try:
                    frames_analyzed += 1
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

                    # Detectar poses e atividades
                    results = pose.process(rgb_frame)
                    activity = detect_activity(results.pose_landmarks, mp_pose)

                    # Identificar anomalias
                    is_anomalous = activity == "Nao identificado ou anomalo"
                    if is_anomalous:
                        anomalies_detected += 1

                    # Adicionar resultados do frame
                    frame_result = {
                        "frame": frames_analyzed,
                        "num_faces": len(face_locations),
                        "emotions": emotions,
                        "activity": activity,
                    }
                    results_data.append(frame_result)

                    # Desenhar resultados no frame
                    for (top, right, bottom, left), emotion in zip(face_locations, emotions):
                        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                        cv2.putText(frame, emotion, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    cv2.putText(frame, f"Atividade: {activity}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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

        # Adicionar resumo ao relatório
        results_summary = {
            "total_frames": total_frames,
            "frames_analyzed": frames_analyzed,
            "anomalies_detected": anomalies_detected,
            "frames_data": results_data
        }

        # Salvar relatório completo no JSON
        with open(output_json_path, "w", encoding="utf-8") as json_file:
            json.dump(results_summary, json_file, indent=4, ensure_ascii=False)

        print(f"Resultados salvos em: {output_json_path}")


def analyze_emotions(frame, face_locations):
    """
    Detectar emoções em cada rosto encontrado no frame.
    """
    emotions = []
    for top, right, bottom, left in face_locations:
        try:
            top = max(0, top)
            right = min(frame.shape[1], right)
            bottom = min(frame.shape[0], bottom)
            left = max(0, left)

            face_region = frame[top:bottom, left:right]
            if face_region.size == 0:
                emotions.append("Indefinido")
                continue

            result = DeepFace.analyze(face_region, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0].get('dominant_emotion', "Indefinido") if isinstance(result, list) else result.get('dominant_emotion', "Indefinido")
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
        # Landmarks principais
        left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        left_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_eye = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
        right_eye = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]
        left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        mouth = pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT]


         # Identificar movimentos
        circular_motion = (
            abs((left_wrist.x - left_elbow.x) * (left_elbow.y - left_shoulder.y) -
                (left_elbow.x - left_shoulder.x) * (left_wrist.y - left_elbow.y)) > 0.05
        )
        arm_movement = (
            abs(left_wrist.y - left_elbow.y) > 0.3 or abs(right_wrist.y - right_elbow.y) > 0.3
        )
        leg_movement = (
            abs(left_ankle.y - left_knee.y) > 0.3 or abs(right_ankle.y - right_knee.y) > 0.3
        )
        leg_displacement = (
            abs(left_knee.x - right_knee.x) > 0.2 or abs(left_ankle.x - right_ankle.x) > 0.2
        )
        alternating_leg_motion = (
            abs(left_knee.y - left_ankle.y) > 0.2 and abs(right_knee.y - right_ankle.y) > 0.2
        )
        shoulder_alignment = abs(left_shoulder.y - right_shoulder.y) < 0.3
        hip_alignment = abs(left_hip.y - right_hip.y) < 0.1
        facial_movement = (
            abs(left_eye.y - right_eye.y) > 0.05 and  # Diferença de altura entre os olhos
            abs(mouth.y - nose.y) > 0.1 and          # Movimentos significativos da boca em relação ao nariz
            abs(left_eye.x - right_eye.x) < 0.05     # Distância mínima entre olhos
        )
        hand_vertical_movement = (
            abs(left_wrist.y - left_elbow.y) > 0.2 or abs(right_wrist.y - right_elbow.y) > 0.2
        )
        # Identificar mãos acima dos ombros
        hands_above_shoulders = (
            left_wrist.y < left_shoulder.y or right_wrist.y < right_shoulder.y
        )
        # Identificar alinhamento horizontal para "Deitado"
        eye_alignment = abs(left_eye.y - right_eye.y) < 0.05  # Olhos alinhados horizontalmente
        mouth_alignment = abs(mouth.y - nose.y) < 0.05  # Boca alinhada horizontalmente com o nariz
        shoulder_alignment_horizontal = abs(left_shoulder.y - right_shoulder.y) < 0.1
        hip_alignment_horizontal = abs(left_hip.y - right_hip.y) < 0.1

        
        # Dançando: pernas e braços em movimento simultâneo
        if leg_movement and arm_movement and circular_motion and not shoulder_alignment:
            return "Dancando"

        if hands_above_shoulders and hand_vertical_movement:
            return "Maos para cima"
        
        # Andando: movimento alternado das pernas
        if leg_displacement and alternating_leg_motion and not shoulder_alignment:
            return "Andando"

        # Fazendo careta: movimentos estranhos da boca e olhos
        if facial_movement:
            return "Fazendo Careta"

        # Escrevendo: movimento leve da mão (vertical), sem movimentos laterais significativos
        if abs(left_wrist.y - left_elbow.y) > 0.1 and abs(right_wrist.y - right_elbow.y) < 0.05:
            return "Escrevendo"
        
    # Movimento não identificado ou anômalo
    return "Nao identificado ou anomalo"


# Configurações do vídeo de entrada e saída
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'video_teste.mp4')
output_video_path = os.path.join(script_dir, 'output_video_otimizado_final_1.mp4')

# Processar o vídeo
detect_pose_and_faces(input_video_path, output_video_path)
