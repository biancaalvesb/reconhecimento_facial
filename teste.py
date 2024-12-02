import cv2
import face_recognition
from deepface import DeepFace
import mediapipe as mp
import pandas as pd

# Inicialize os modelos fora do loop
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
data = []

# Função para localizar faces
def recognize_faces(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    return face_locations

# Função para desenhar retângulos ao redor das faces detectadas
def draw_faces(frame, face_locations):
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    return frame

# Função para analisar emoções faciais
def analyze_emotions(frame, face_locations):
    results = []
    for i, (top, right, bottom, left) in enumerate(face_locations):
        if i >= 1:  # Limitar a análise a 1 face por frame
            break
        face = frame[top:bottom, left:right]
        try:
            result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            results.append(result['dominant_emotion'])
        except:
            results.append("Indefinido")
    return results

# Função para detectar atividades usando MediaPipe
def detect_activities(frame):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return results

# Função para gerar relatório
def generate_report(data, output_path="report.csv"):
    df = pd.DataFrame(data)
    print(df.head())  # Mostrar primeiras linhas no terminal
    df.to_csv(output_path, index=False)
    print(f"Relatório salvo em: {output_path}")

# Função principal de captura de vídeo
def capture_video():
    cap = cv2.VideoCapture("video_teste.mp4")
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Configurar o gravador de vídeo (VideoWriter)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define o codec e cria o objeto VideoWriter
    out = cv2.VideoWriter(
        'video_processado_2.mp4',
        cv2.VideoWriter_fourcc(*'MP4V'),
        fps,
        (frame_width, frame_height)
    )

    total_frames = 0
    intervalo_entre_frames = 30

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Fim do vídeo ou erro ao ler frame.")
                break

            total_frames += 1

            # Processar apenas frames em intervalos definidos
            if total_frames % intervalo_entre_frames == 0:
                try:
                    face_locations = recognize_faces(frame)
                    emotions = analyze_emotions(frame, face_locations)
                    results = detect_activities(frame)
                    activities = "Dançando" if results.pose_landmarks else "Indefinido"

                    # Adicionar dados ao relatório
                    data.append({
                        "frame": total_frames,
                        "faces": len(face_locations),
                        "emotions": emotions,
                        "activities": activities
                    })

                    # Desenhar rostos detectados
                    frame = draw_faces(frame, face_locations)
                except Exception as e:
                    print(f"Erro no processamento: {e}")

            # Gravar o frame processado
            out.write(frame)

            # Mostrar o frame processado na tela
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Execução interrompida pelo usuário.")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

# Executa o script
if __name__ == "__main__":
    capture_video()
    generate_report(data)
