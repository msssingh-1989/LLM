### Objective: 
To prepare the diagonsis report summary in specified format from given input text to find the key symptoms, Vital signs, Allergies and Laboratory tests for automation and Electronic health record analysis purpose

### Approach:
- Use OpenSource Large language Model to extract the required information from input text 
- In order to make the LLM align with Medical and diagonsis domain , we will use few-shot prompt learning by giving few sample example from train dataset with input and output
- Create a custom prompt with requested input with Few shot learning and 
- Use OpenAI GPT LLM (or Other Opensource LLM) model for text summarization in prescribed format 
    - #### Note: Other Opensource LLM models such as Falcon-40B can be used in-premise for better data security but need more compute resources such as GPUs
- Create LLM chain i.e. pipeline to process raw input into Prompt , LLM model response and Output response post processing in batch mode
- Develop an API to get the input texts as input from user and provide the reponse 

### LLM Model response Evaluation 
- Metric: "ROUGE" that measure the overlap and similarity between the generated summary and a reference summary that serves as a benchmark.
- ROUGE-L: This metric evaluates the match of the longest common subsequence of words between the two texts.
- Below is summary for 2 Validation input text response compared with actual report summary after few-shot learning
- Metric performance summary:

    |    Metric    |   Value    |
    |-----------|---------|
    | 'rouge1' | 0.9255 |
    | 'rouge2' | 0.8669 |
    | 'rougeL' | 0.9184 |
    | 'rougeLsum' | 0.9255 |





#### Get the source code 
- Clone the repo using below command
    ```bash 
        git clone 
    ```
- Change the dir to project dir i.e. "LLM"

#### Environment Setup:
- Install Python = 3.10
- Install all the dependencies using below command
    ```bash
        pip install -r requirements.txt
    ```

#### Pre-requisite:
- Rename file ".example.env" to ".env"
- Update the "open_ai_secret_key" .
    - Click on link https://platform.openai.com/api-keys
    - Generate secret key as highlighted in below snapshot
    - image_url: docs\openai_developer_APIKey.png
    - Update the "API_key" in ".env" file after generating the secret key from above step



#### RUN the API
- Command
```python 
        uvicorn main:app 
```
- Open the browser and go to below link
- FAST API url : http://127.0.0.1:8000/docs/


#### API Sample Request and Response in batch mode:

- Sample CURL POST Request
```python
curl --location 'http://127.0.0.1:8000/report_summary/' \
--header 'Content-Type: application/json' \
--data '{
    "input_text": [
        "26-year-old young adult; reports frequent sicknesss and fatigue with breathing issue. cholestrol levels are 65-85; platellates are below 45 mu/l. Chest CT scans are indicating mid stage lung cancers and pneumonia. Suggested high dose antibioticss to monitor oxygen levels and schedule a follow-up for surgery."
        ,"39-year-old young adult; reports frequent sicknesss and fatigue with breathing issue. cholestrol levels are 65-85; platellates are below 45 mu/l. Chest CT scans are indicating mid stage lung cancers and pneumonia. Suggested high dose antibioticss to monitor oxygen levels and schedule a follow-up for surgery."
    ]
}'

```
- Sample Response

```json

[
    {
        "report_input": "26-year-old young adult; reports frequent sicknesss and fatigue with breathing issue. cholestrol levels are 65-85; platellates are below 45 mu/l. Chest CT scans are indicating mid stage lung cancers and pneumonia. Suggested high dose antibioticss to monitor oxygen levels and schedule a follow-up for surgery.",
        
        "predicted_report_output": "\n 26-year-old young adult presenting with frequent sickness and fatigue with breathing issues.\nVital signs within normal ranges.\nAllergies: None reported.\nImmunizations up to date.\nLaboratory tests indicate cholesterol levels between 65-85 and platellates below 45 mu/l.\nChest CT scans indicate mid stage lung cancer and pneumonia.\nNo remarkable findings in the cardiovascular and respiratory system examination.\nGastrointestinal and neurological examinations show no abnormalities.\nSuggests high dose antibiotics to monitor oxygen levels and schedule a follow-up for surgery."
    },
    {
        "report_input": "39-year-old young adult; reports frequent sicknesss and fatigue with breathing issue. cholestrol levels are 65-85; platellates are below 45 mu/l. Chest CT scans are indicating mid stage lung cancers and pneumonia. Suggested high dose antibioticss to monitor oxygen levels and schedule a follow-up for surgery.",
        
        "predicted_report_output": "\n 39-year-old young adult presenting with frequent sickness and fatigue with breathing issues.\nVital signs within normal ranges.\nAllergies: None reported.\nImmunizations up to date.\nLaboratory tests indicate cholesterol levels between 65-85 and platelet count below 45 mu/l.\nChest CT scans indicate mid stage lung cancer and pneumonia.\nNo remarkable findings in the cardiovascular and respiratory system examination.\nGastrointestinal and neurological examinations show no abnormalities.\nSuggests high dose antibiotics to monitor oxygen levels and schedule a follow-up for surgery."
    }
]

```