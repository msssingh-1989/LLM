import os


from src.prescription.llm_report import *
from fastapi import FastAPI
from langserve import add_routes
import json


from fastapi import FastAPI
from fastapi import Request

from pydantic import BaseModel, validator
from fastapi.encoders import jsonable_encoder

class Input_request(BaseModel):
    """ API Input Requests 
    """
    input_text: list


app = FastAPI()


@app.post("/report_summary/")
def report_summary(input_request:Input_request):
    """ - FAST API with post request takes list of texts as input, convert in specific format using prompt template 
        - Process the prompt with input text through LLM Chain pipeline (prompts+LLM model) and provide the response in prescribed format
    """

    input_request = jsonable_encoder(input_request)

    test_report_input = input_request["input_text"]

    llm_report_obj = llm_report(input_text="train.xlsx",file_modeYN=1)

    llm_report_obj.train_examples_df,llm_report_obj.test_examples_df = llm_report_obj.f_fetch_examples_from_file_data(n_shot_example=2)

    prompt = llm_report_obj.f_generate_few_shot_prompt_template()

    llm_report_obj.llm_chain = llm_report_obj.f_get_llm_chain()

    result,llm_report_obj.test_examples_df = llm_report_obj.f_evaluate_llm_response()

    test_report_output = llm_report_obj.f_get_live_llm_response(test_report_input)


    return test_report_output


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)