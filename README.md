# Tech Challenge - Análise de Vídeo com Reconhecimento Facial, Expressões e Atividades

Grupo 5
Bianca Alves Barroso          - RM355265
Carlos Kuchenbecker da Silva  - RM356081
João Amaro Alves Pessanha     - RM355800
Silvanio dos Santos Ferreira  - RM354804

## Descrição do Projeto
Este projeto foi desenvolvido como parte do **Tech Challenge - Fase 4** e utiliza técnicas avançadas de análise de vídeo para realizar as seguintes tarefas:
1. **Reconhecimento Facial:** Identifica e marca os rostos presentes no vídeo.
2. **Análise de Expressões Emocionais:** Detecta e categoriza as emoções de cada rosto identificado.
3. **Detecção de Atividades:** Classifica e categoriza atividades realizadas no vídeo, incluindo "Dançando", "Andando", "Acenando", "Fazendo Careta", entre outras.
4. **Identificação de Anomalias:** Detecta movimentos fora do padrão geral das atividades esperadas.
5. **Geração de Resumo:** Cria um relatório JSON com o total de frames analisados, número de anomalias detectadas e as atividades realizadas.

## Funcionalidades
- **Reconhecimento Facial:** Utiliza o MediaPipe para detecção de rostos e DeepFace para análise de emoções.
- **Detecção de Poses e Atividades:** Implementada com MediaPipe Pose para identificar movimentos e classificá-los.
- **Marcação no Vídeo:** Os rostos e atividades detectados são marcados diretamente no vídeo, exibindo informações em tempo real.
- **Resumo Automático:** Gera um arquivo JSON com um resumo do vídeo, incluindo:
  - Total de frames analisados.
  - Número de anomalias detectadas.
  - Atividades e emoções identificadas.

## Estrutura do Projeto
- `detect_pose_and_faces(video_path, output_path)`: Função principal para processar o vídeo.
- `analyze_emotions(frame, face_locations)`: Detecta emoções nos rostos identificados.
- `detect_activity(pose_landmarks, mp_pose)`: Classifica as atividades com base nos landmarks corporais.

## Requisitos
Para executar o projeto, é necessário ter as seguintes dependências instaladas:
- Python 3.9 ou superior
- OpenCV
- Face Recognition
- DeepFace
- MediaPipe
- tqdm
- NumPy
- TensorFlow

### Instalação de Dependências
Você pode instalar todas as dependências com o seguinte comando:
pip install opencv-python face_recognition deepface mediapipe tqdm numpy tensorflow

### Como Executar
- Clone este repositório:

git clone <URL_DO_REPOSITORIO>
cd <PASTA_DO_REPOSITORIO>

- Coloque o vídeo de entrada na pasta principal e nomeie-o como video_teste.mp4.
- Execute o script principal: 
python facial_recognition.py
- O vídeo processado será salvo na pasta output com o nome output_video_otimizado.mp4, junto com o arquivo results.json contendo o resumo.