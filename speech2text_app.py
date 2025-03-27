import torch
from transformers import pipeline
import gradio as gr

# OpenAI Whisper 모델을 사용하여 오디오를 전사하는 함수
def transcript_audio(audio_file):
    # 음성 인식 파이프라인 초기화
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
    )
    # 오디오 파일을 전사하고 결과 반환
    result = pipe(audio_file, batch_size=8)["text"]
    
    return result
    # Send the transcribed text to the llm
    transcribed_text = result
    llm_response = llm(transcribed_text)  # Assuming llm is accessible here
    
    return transcribed_text + "\n\nLLM Response:\n" + llm_response
    
# Gradio 인터페이스 설정
audio_input = gr.Audio(sources="upload", type="filepath")  # 오디오 입력
output_text = gr.Textbox()  # 텍스트 출력

# 함수, 입력 및 출력으로 Gradio 인터페이스 생성
iface = gr.Interface(fn=transcript_audio, 
                     inputs=audio_input, outputs=output_text, 
                     title="오디오 전사 앱",
                     description="오디오 파일을 업로드하세요")

# Gradio 앱 실행
iface.launch(server_name="0.0.0.0", server_port=7860,share=True)