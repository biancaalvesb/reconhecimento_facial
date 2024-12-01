import cv2

def capture_video():
    # Iniciar a captura de vídeo
    cap = cv2.VideoCapture("video_teste.mp4")
    print("Acessando o vídeo")

    if not cap.isOpened():
        print("Erro ao acessar o vídeo.")
        return

    # Configurar o gravador de vídeo (VideoWriter)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define o codec e cria o objeto VideoWriter
    out = cv2.VideoWriter(
        'video_processado.mp4',  # Nome do arquivo de saída
        cv2.VideoWriter_fourcc(*'MP4V'),
        fps,  # FPS do vídeo
        (frame_width, frame_height)  # Tamanho dos frames
    )

    # Carregar o classificador Haar Cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    if face_cascade.empty():
        print("Erro ao carregar o classificador Haar Cascade.")
        return

    total_frames = 0
    intervalo_entre_frames = 30

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Fim do vídeo ou erro ao ler frame.")
                break

            # Incrementa o contador de frames
            total_frames += 1

            # Realiza a análise somente dos frames especificados
            if total_frames % intervalo_entre_frames == 0:
                # Conversão para escala de cinza
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Pré-processamento
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                gray = cv2.equalizeHist(gray)

                # Detecção de rostos
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(50, 50))

                # Log de rostos detectados
                print(f"[Frame {total_frames}] Detectado {len(faces)} rosto(s).")

                # Desenhar retângulos ao redor dos rostos
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Sempre exibir o frame
            cv2.imshow('Face Detection', frame)

            # Gravar o frame processado no vídeo de saída
            out.write(frame)

            # Encerrar ao pressionar 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("Encerrando...")
                break

    except KeyboardInterrupt:
        print("Execução interrompida pelo usuário.")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
    finally:
        # Liberar recursos
        cap.release()
        out.release()  # Liberar o VideoWriter
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_video()
