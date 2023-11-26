import os
from dotenv import load_dotenv

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from operator import itemgetter


from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import ChatPromptTemplate

from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch

from langchain.llms import OpenAI,HuggingFaceHub
from langchain.chat_models import ChatOpenAI

import evaluate

from src.common.path_setup import *


config_env = load_dotenv(".env")



class llm_report():
    """
    Class: 
        - Encompass all the functionality to fine tune LLM model (OPENAI - GPT) using N-shot prompt fine tuning 
        - Create langchain with prompt and LLM model to create the pipeline
        - Live API request, with LLM response
    """
    def __init__(self,input_text:str="train.xlsx",file_modeYN:int=1,eval_modeYN:int=0):
        """ Initialize working directory with input file path setup
        """

        self.work_dir = work_dir
        self.input_dir = input_dir
        self.output_dir = output_dir

        self.file_modeYN = file_modeYN
        
        if self.file_modeYN == 1 :
            self.input_file_name = input_text

    def f_fetch_examples_from_file_data(self,n_shot_example=2):
        """
        Objective: 
            - Read the input file and prepare data as training and validation for few shot prompt finetuning 
        Parameters:
            - n_shot_example: default = 2
                Utilize the input to decide the sample no. of example to be used in prompt fine-tuning
        """
        
        if self.file_modeYN == 1 :
            input_df = pd.read_excel(os.path.join(self.input_dir,self.input_file_name))  ##,encoding = enc['encoding']

        train, test = train_test_split(input_df,random_state=42,test_size=0.3)

        train_examples_df  = train.sample(np.min([n_shot_example,train.shape[0]]))
        test_examples_df  = test.sample(np.min([n_shot_example,test.shape[0]]))

        return train_examples_df,test_examples_df


    @staticmethod
    def f_format_examples_to_prompt(examples_df,prompt_vars = ["report_input","report_output"]):
        """
        Objective: 
            - Read the given input text and covert it to map the input data in the sample prompt 
        Parameters:
            - examples_df: 
                Any example of text input to be mapped to input in the sample prompt

        
        """
                
        examples_df.columns = prompt_vars

        return examples_df.to_dict(orient = 'records')
    

    def f_generate_few_shot_prompt_template(self):
        """
        Objective: 
            - Create custom prompt template with instructio to LLM and few shot example with expected output 

        
        """

        example_prompt_template = """ 
                    You are a experienced doctor and analysing the symptoms, diagonstic reports of a patient basis given context. 
                    After Analysing the complete context, provide the final report in prescribed format. Below are few samples.

                    input report: {report_input}
                    output report: {report_output}

                    """


        example_prompt = PromptTemplate(input_variables=["report_input", "report_output"], template=example_prompt_template)


        
        ### few shots
        examples = self.f_format_examples_to_prompt(examples_df=self.train_examples_df,prompt_vars = ["report_input","report_output"])

        



        prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            suffix="input report: {report_input}",
            input_variables=["report_input"]
        )

        return prompt
    

    def f_get_llm_chain(self):
        """
        Objective: 
            - Create LLM chain pipeline using Prompts + llm MODEL For inference

        """

        ## chatgpt
        model = OpenAI(temperature=0,openai_api_key=os.getenv("open_ai_secret_key"))

        ## prompt
        prompt = self.f_generate_few_shot_prompt_template()

        # #Create a chain using interface to get LLM Response
        llm_chain = prompt | model

        return llm_chain
    
    
    def f_evaluate_llm_response(self):
        """
        Objective: 
            - Evaluate the LLM response on validation sample input examples  and assess performance using metric "ROUGE"
        """
        test_examples = self.f_format_examples_to_prompt(examples_df=self.test_examples_df,prompt_vars = ["report_input","report_output"])
        test_report_input_all = [{"report_input": x["report_input"]} for x in test_examples]

        batch_output = self.llm_chain.batch(test_report_input_all, config={"max_concurrency": 5})

        self.test_examples_df["predicted_report_output"] = batch_output
        self.test_examples_df["predicted_report_output"] = self.test_examples_df["predicted_report_output"].str.replace("output report:","")

        self.test_examples_df.to_excel(os.path.join(self.output_dir,f"""output_{self.input_file_name}"""))

        ### model evaluate

        rouge = evaluate.load('rouge')
        # blue = evaluate.load('blue')

        metric_results = rouge.compute(predictions=self.test_examples_df["predicted_report_output"].tolist(),
                                references=self.test_examples_df["report_output"].tolist())
        
        return metric_results, self.test_examples_df
    

    def f_get_live_llm_response(self,test_report_input:list=[""]):
        """
        Objective: 
            - Read the given input text and pass it to LLM Chain to get the response on live request in batch mode through API 
        Parameters:
            - test_report_input: 
                List of example text input 

        
        """

        ### test data model output
        # test_report_input  = """60-year-old female; reports frequent headaches and blurred vision. BP reads 130/85; blood sugar levels slightly elevated. Ophthalmologist notes signs of early cataract formation. Advised to monitor blood sugar levels and schedule a follow-up for cataract assessment.
        # """

        test_report_input_sample = [{"report_input": x} for x in test_report_input]



        batch_output = self.llm_chain.batch(test_report_input_sample, config={"max_concurrency": 5})

        batch_output_df = pd.DataFrame.from_dict({"report_input":test_report_input,
                                                "predicted_report_output":batch_output})
        batch_output_df["predicted_report_output"] = batch_output_df["predicted_report_output"].str.replace("output report:","")


        return batch_output_df.to_dict(orient="records")   

    























