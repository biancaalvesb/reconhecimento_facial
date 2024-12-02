import cv2
import face_recognition
from deepface import DeepFace
import mediapipe as mp
import os
from tqdm import tqdm
import json


def detect_pose_and_faces(video_path, output_path):
    """
    Detecta poses, rostos e emoções no vídeo.
    Salva os resultados em um arquivo JSON por frame.
    """
    # Inicializar MediaPipe Pose
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

    # Criar a pasta "output" se não existir
    output_dir = os.path.join(os.path.dirname(output_path), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Atualizar o caminho de saída para a pasta "output"
    output_video_path = os.path.join(output_dir, os.path.basename(output_path))
    output_json_path = os.path.join(output_dir, "results.json")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_skip = 5  # Processar 1 em cada 5 frames
    results_data = []  # Para salvar os resultados

    try:
        for frame_count in tqdm(range(total_frames), desc="Processando vídeo"):
            ret, frame = cap.read()
            if not ret:
                break

            # Processar apenas frames em intervalos definidos
            if frame_count % frame_skip == 0:
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Detectar rostos
                    face_locations = face_recognition.face_locations(rgb_frame, model='cnn')

                    # Analisar emoções
                    emotions = analyze_emotions(frame, face_locations)

                    # Desenhar retângulos e exibir emoções
                    for (top, right, bottom, left), emotion in zip(face_locations, emotions):
                        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                        cv2.putText(frame, emotion, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        print(f"Face coordinates: top={top}, right={right}, bottom={bottom}, left={left}")

                    # Detectar poses e atividades
                    results = pose.process(rgb_frame)
                    activity = detect_activity(results.pose_landmarks, mp_pose)

                    # Adicionar atividade no vídeo
                    cv2.putText(frame, f"Atividade: {activity}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Desenhar landmarks de pose
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # Salvar os resultados do frame
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

        # Salvar resultados no arquivo JSON
        with open(output_json_path, "w", encoding="utf-8") as json_file:
            json.dump(results_data, json_file, indent=4, ensure_ascii=False)

        print(f"Resultados salvos em: {output_json_path}")


def analyze_emotions(frame, face_locations):
    """
    Detectar emoções em cada rosto encontrado no frame.
    """
    emotions = []
    for (top, right, bottom, left) in face_locations:
        # Garantir que os índices estão dentro dos limites
        top = max(0, top)
        right = min(frame.shape[1], right)
        bottom = min(frame.shape[0], bottom)
        left = max(0, left)
        
        face = frame[top:bottom, left:right]
        
        # Verificar se o recorte não está vazio
        if face.size == 0:
            emotions.append("Indefinido")
            continue
        
        try:
            result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result['dominant_emotion']

            # Mapear emoções detectadas para categorias específicas
            if dominant_emotion == 'happy':
                emotions.append("Sorrindo")
            elif dominant_emotion == 'sad':
                emotions.append("Chorando")
            elif dominant_emotion == 'surprise':
                emotions.append("Assustado/Surpreso")
            elif dominant_emotion == 'angry':
                emotions.append("Bravo")
            elif dominant_emotion == 'fear':
                emotions.append("Medo")
            elif dominant_emotion in ['neutral']:
                emotions.append("Neutro")
            else:
                emotions.append("Indefinido")

            print(f"Emotion detected: {dominant_emotion}")
        except Exception as e:
            print(f"Erro na análise de emoções: {e}")
            emotions.append("Indefinido")
    return emotions


def detect_activity(pose_landmarks, mp_pose):
    """
    Detectar atividades com base em landmarks de pose.
    """
    if pose_landmarks:
        left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        left_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]

        # Acenando
        if abs(left_wrist.y - left_elbow.y) > 0.1 or abs(right_wrist.y - right_elbow.y) > 0.1:
            return "Acenando"

        # # Escrevendo
        # if left_wrist.x < left_elbow.x or right_wrist.x > right_elbow.x:
        #     return "Escrevendo"

        # Dançando
        if (abs(left_wrist.y - left_elbow.y) > 0.1 and abs(left_ankle.y - left_knee.y) > 0.1) or \
           (abs(right_wrist.y - right_elbow.y) > 0.1 and abs(right_ankle.y - right_knee.y) > 0.1):
            return "Dançando"

        # Andando
        if abs(left_ankle.x - right_ankle.x) > 0.1 and abs(left_knee.y - right_knee.y) > 0.1:
            return "Andando"

    return "Indefinido"


# Configurações do vídeo de entrada e saída
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'video_teste.mp4')
output_video_path = os.path.join(script_dir, 'video_processado_emocoes_atividades.mp4')

# Processar o vídeo
detect_pose_and_faces(input_video_path, output_video_path)
