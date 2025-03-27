import torch
import os
import gradio as gr

from langchain.llms import OpenAI
from langchain.llms import HuggingFaceHub

from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

my_credentials = {
    "url"    : "https://us-south.ml.cloud.ibm.com"
}
params = {
        GenParams.MAX_NEW_TOKENS: 800, # 모델이 한 번의 실행에서 생성할 수 있는 최대 토큰 수.
        GenParams.TEMPERATURE: 0.1,   # 토큰 생성의 무작위성을 조절하는 파라미터. 낮은 값은 생성이 더 결정론적이게 만들고, 높은 값은 더 많은 무작위성을 도입합니다.
    }

LLAMA2_model = Model(
        model_id= 'meta-llama/llama-3-2-11b-vision-instruct', 
        credentials=my_credentials,
        params=params,
        project_id="skills-network",  
        )

llm = WatsonxLLM(LLAMA2_model)  

#######------------- 프롬프트 템플릿-------------####

temp = """
<s><<SYS>>
문맥에서 세부 사항과 함께 핵심 사항을 나열하세요: 
[INST] 문맥 : {context} [/INST] 
<</SYS>>
"""

pt = PromptTemplate(
    input_variables=["context"],
    template= temp)

prompt_to_LLAMA2 = LLMChain(llm=llm, prompt=pt)

#######------------- 음성 텍스트 변환-------------####

def transcript_audio(audio_file):
    # 음성 인식 파이프라인 초기화
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
    )
    # 오디오 파일을 전사하고 결과를 반환
    transcript_txt = pipe(audio_file, batch_size=8)["text"]
    result = prompt_to_LLAMA2.run(transcript_txt)

    return result

#######------------- 그라디오-------------####

audio_input = gr.Audio(sources="upload", type="filepath")
output_text = gr.Textbox()

iface = gr.Interface(fn= transcript_audio, 
                    inputs= audio_input, outputs= output_text, 
                    title= "오디오 전사 앱",
                    description= "오디오 파일을 업로드하세요")

iface.launch(server_name="0.0.0.0", server_port=7860)