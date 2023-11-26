import os
from dotenv import load_dotenv

import pandas as pd
import numpy as np

from langchain.llms import OpenAI,HuggingFaceHub
from langchain.chat_models import ChatOpenAI


config_env = load_dotenv(".env")



####### 


llm = OpenAI(openai_api_key=os.getenv("open_ai_secret_key"))
chat_model = ChatOpenAI(openai_api_key=os.getenv("open_ai_secret_key"))




############## using template 
from langchain import PromptTemplate

report_input  = """

I bought last week the latest Samsung washing machine. 
The product is too expensive for what it is. 
I won't recommend it to anyone.

"""

# llm = OpenAI(model_name="text-davinci-003")

template = """ 
You are a experienced doctor and analysing the symptoms, diagonstic reports of a patient basis given context. 
After Analysing the complete reports provide the final output is prescribed format.

For any new report_input that is sent to you, extract the following information:

Symptoms: What critical symptoms and signs patient have?\
Answer the key symptoms. If this information is not found, output unknown.

Vital signs: What are different lab test reports such as blood pressure, sugar levels, blood test reports etc?
Answer the key signal from lab test reports. If this information is not found, output unknown. 

Allergies: Does patient have any kind of allergies and different levels? \
Answer allergies related details, if this information is not found, output None reported.

Remark: List down other key findings from patient reports such as X-ray, ECG, medical reports recommendations, follow ups? \
Answer all remaining important report summary , if this information is not found, output unknown.


Format the output as JSON with the following keys:
Symptoms
Vital signs
Allergies
Remark

diagnosis report: {report_input}

"""

prompt = PromptTemplate(

input_variables=["report_input"],

template=template,

)

final_prompt = prompt.format(report_input=report_input)

print(f"LLM Output: {llm(final_prompt)}")





##### output parser

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

response_schemas = [
    
    ResponseSchema(name="Symptoms", description="Answer the key symptoms."),
    ResponseSchema(name="Vital signs", description="Answer the key signal from lab test reports"),
    ResponseSchema(name="Allergies",       description="Answer allergies related details"),
    ResponseSchema(name="Remark",       description="Answer all remaining important report summary.")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()

template = """ 
You are a report_input checker specialist. You main goal is to extract worth information from any report_input that is sent to you. 
From the following report_input {report_input}, extract the following information:
{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["report_input"],
    partial_variables={"format_instructions": format_instructions}
)




##### model and reponse 
model = OpenAI(temperature=0,openai_api_key=os.getenv("open_ai_secret_key"))
# model = HuggingFaceHub(repo_id="mistralai/Mistral-7B-v0.1", huggingfacehub_api_token=HF_API_KEY)

_input = prompt.format_prompt(report_input=report_input)
output = model(_input.to_string())
output_json = output_parser.parse(output)
output_json