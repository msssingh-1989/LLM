import os
from dotenv import load_dotenv

import pandas as pd
import numpy as np
# from prescription import work_dir,input_dir

from sklearn.model_selection import train_test_split
from operator import itemgetter

# from langchain import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import ChatPromptTemplate

from langchain.llms import OpenAI,HuggingFaceHub
from langchain.chat_models import ChatOpenAI


from common.path_setup import *


config_env = load_dotenv(".env")




####### 


# llm = OpenAI(openai_api_key=os.getenv("open_ai_secret_key"))
# chat_model = ChatOpenAI(openai_api_key=os.getenv("open_ai_secret_key"))




############## using template 




######### Read sample input

input_file_name = "train.xlsx"

import chardet
import pandas as pd

# with open(os.path.join(input_dir,input_file_name), 'rb') as f:
#     enc = chardet.detect(f.read())

input_df = pd.read_excel(os.path.join(input_dir,input_file_name))  ##,encoding = enc['encoding']

train, test = train_test_split(input_df,random_state=42,test_size=0.3)

n_shot_example = 2

train_examples_df  = train.sample(np.min([n_shot_example,train.shape[0]]))
test_examples_df  = test.sample(np.min([5,test.shape[0]]))

def f_format_examples_to_prompt(examples_df,prompt_vars = ["report_input","report_output"]):
    examples_df.columns = prompt_vars

    return examples_df.to_dict(orient = 'records')


examples = f_format_examples_to_prompt(train_examples_df,prompt_vars = ["report_input","report_output"])
test_examples = f_format_examples_to_prompt(test_examples_df,prompt_vars = ["report_input","report_output"])


# examples = [
# #   {
# #     "report_input": "45 year old male ; complains of intermittent chest pain and shortness of breath. BP reads 140/90 ; cholesterol levels are high ; Slight irregularities are noted in ECG. Cardiologist suggests stress test, Chest Xray and recommends dietary changes. ",
# #     "report_output":
# # """
# # 45-year-old male presenting with intermittent chest pain and shortness of breath.
# # Vital signs: Blood pressure slightly elevated at 140/90.
# # Allergies: None reported.
# # Immunizations up to date.
# # Laboratory tests indicate high cholesterol levels. ECG shows slight irregularities.
# # No remarkable findings in the cardiovascular and respiratory system examination.
# # Gastrointestinal and neurological examinations show no abnormalities.
# # Cardiologist suggests a stress test, Chest Xray and recommends dietary changes. 
# # """
# #   },
#   {
#     "report_input": "30-year-old female; complains of frequent urination and extreme thirst. History of diabetes in family. Blood sugar level is 200 mg/dL.Recommends further evaluation for diabetes management. ",
#     "report_output":
# """
# 30-year-old female experiencing frequent urination and extreme thirst. History of diabetes in family. 
# Vital signs within normal ranges.
# Allergies: None reported.
# Immunizations up to date.
# Blood sugar level significantly elevated at 200 mg/dL, suggesting possible diabetes.
# No remarkable findings in the cardiovascular and respiratory system examination.
# Gastrointestinal and neurological examinations show no abnormalities.
# Recommends further evaluation for diabetes management. 
# """
#   }
#   ,
# {
#     "report_input": "42-year-old male; routine check-up. BP reads 120/80; no complaints of any symptoms. All recent laboratory tests are within normal limits. No history of chronic illnesses.",
#     "report_output":
# """
# 42-year-old male presenting for a routine check-up. No complaints of any symptoms.No history of chronic illnesses.
# Vital signs: Blood pressure normal at 120/80.
# Allergies: None reported.
# Immunizations up to date.
# All recent laboratory tests fall within normal parameters.
# No remarkable findings in the cardiovascular and respiratory system examination.
# Gastrointestinal and neurological examinations show no abnormalities.
# """
#   }
# ]
# # llm = OpenAI(model_name="text-davinci-003")

example_prompt_template = """ 
You are a experienced doctor and analysing the symptoms, diagonstic reports of a patient basis given context. 
After Analysing the complete context, provide the final report in prescribed format. Below are few samples.

input report: {report_input}
output report: {report_output}

"""


example_prompt = PromptTemplate(input_variables=["report_input", "report_output"], template=example_prompt_template)

print(example_prompt.format(**examples[0]))


prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="input report: {report_input}",
    input_variables=["report_input"]
)

### test data model output
test_report_input  = """60-year-old female; reports frequent headaches and blurred vision. BP reads 130/85; blood sugar levels slightly elevated. Ophthalmologist notes signs of early cataract formation. Advised to monitor blood sugar levels and schedule a follow-up for cataract assessment.
"""

test_actual_report_output = """60-year-old female presenting with frequent headaches and blurred vision.
Vital signs: Blood pressure slightly elevated at 130/85.
Allergies: None reported.
Immunizations up to date.
Laboratory tests show slightly elevated blood sugar levels.
No remarkable findings in the cardiovascular and respiratory system examination.
Gastrointestinal and neurological examinations show no abnormalities.
Ophthalmologist notes early signs of cataract formation.
Advised to monitor blood sugar levels and schedule a follow-up for cataract assessment."""


### model outcome




##### model and reponse 
## chatgpt
model = OpenAI(temperature=0,openai_api_key=os.getenv("open_ai_secret_key"))


## falcon
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch

# model = "tiiuae/falcon-7b-instruct" #tiiuae/falcon-40b-instruct

# tokenizer = AutoTokenizer.from_pretrained(model)

# pipeline = pipeline(
#     "text-generation", #task
#     model=model,
#     tokenizer=tokenizer,
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
#     device_map="auto",
#     max_length=200,
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id
# )

# model = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

# model = HuggingFaceHub(repo_id="tiiuae/falcon-7B", huggingfacehub_api_token=os.getenv("HF_API_KEY"),temperature=0)


test_report_input = [x["report_input"] for x in test_examples]
# test_report_output = [x["report_output"] for x in test_examples]
# final_prompt = prompt.format(report_input=test_report_input[0])

# # print(f"LLM Output: {llm(final_prompt)}")
# _input = final_prompt
# output = model(_input)


# test_model_report_output = """output report:
# 60-year-old female reporting frequent headaches and blurred vision.
# Vital signs: Blood pressure slightly elevated at 130/85.
# Allergies: None reported.
# Immunizations up to date.
# Blood sugar levels slightly elevated.
# Cardiovascular and respiratory examinations reveal no significant abnormalities.
# Ophthalmologic examination indicates early signs of cataract formation.
# Recommends monitoring blood sugar levels and scheduling a follow-up for cataract assessment.
# Gastrointestinal and neurological examinations show no abnormalities."""


# test_model_report_output = output

# #Create a chain using interface to get LLM Response
chain = prompt | model

test_report_input_all = [{"report_input": x["report_input"]} for x in test_examples]

# batch_output = chain.batch([{"report_input": f"""{test_report_input[0]}"""}, {"report_input": f"""{test_report_input[1]}"""}], config={"max_concurrency": 5})


batch_output = chain.batch(test_report_input_all, config={"max_concurrency": 5})

test_examples_df["predicted_report_output"] = batch_output
test_examples_df["predicted_report_output"] = test_examples_df["predicted_report_output"].str.replace("output report:","")

test_examples_df.to_excel(os.path.join(output_dir,f"""output_{input_file_name}"""))
# test_examples_df.to_excel(f"""output_{input_file_name}""")
### model evaluate
import evaluate
rouge = evaluate.load('rouge')
# blue = evaluate.load('blue')

results = rouge.compute(predictions=test_examples_df["predicted_report_output"].tolist(),
                         references=test_examples_df["report_output"].tolist())





# ##### output parser

# from langchain.output_parsers import ResponseSchema
# from langchain.output_parsers import StructuredOutputParser

# response_schemas = [
    
#     ResponseSchema(name="Symptoms", description="report_output the key symptoms."),
#     ResponseSchema(name="Vital signs", description="report_output the key signal from lab test reports"),
#     ResponseSchema(name="Allergies",       description="report_output allergies related details"),
#     ResponseSchema(name="Remark",       description="report_output all remaining important report summary.")
# ]

# output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# format_instructions = output_parser.get_format_instructions()

# output_template = """ 
# You are a report_output checker specialist. You main goal is to extract worth information from any report_input that is sent to you. 
# From the following report_output: {report_output}, extract the following information:
# {format_instructions}
# """

# prompt = PromptTemplate(
#     template=output_template,
#     input_variables=["report_input"],
#     partial_variables={"format_instructions": format_instructions}
# )




# ##### model and reponse 
# model = OpenAI(temperature=0,openai_api_key=os.getenv("open_ai_secret_key"))
# # model = HuggingFaceHub(repo_id="mistralai/Mistral-7B-v0.1", huggingfacehub_api_token=HF_API_KEY)

# _input = prompt.format_prompt(report_input=report_input)
# output = model(_input.to_string())
# output_json = output_parser.parse(output)
# output_json










########### Create an API:

#!/usr/bin/env python
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langserve import add_routes


app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

# add_routes(
#     app,
#     ChatOpenAI(),
#     path="/openai",
# )

# add_routes(
#     app,
#     ChatAnthropic(),
#     path="/anthropic",
# )

# model = ChatAnthropic()
# prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
add_routes(
    app,
    chain,
    path="/report_summary",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)