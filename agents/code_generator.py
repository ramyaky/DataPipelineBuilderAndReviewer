import requests
from dotenv import load_dotenv
import json
import os

### Load environment variables from .env file
load_dotenv()

MODEL = os.getenv("MODEL")
OLLAMA_URL = os.getenv("OLLAMA_URL")

def generate_spark_job(instruction):
    prompt = """
        You are an expert data engineer. 
        Write clean, production ready PySpark code for the following instruction:{instruction}

        Requirements:
        -- Use Spark best practices
        -- Add minimal but clear comments
        -- Do not include example data
        -- Return only valid python code
    """

    llm_result = query_ollama(prompt, MODEL)
    return llm_result

def query_ollama(prompt, model):
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload)

    if response.status_code != 200:
        raise Exception(f"Ollama error {response.text}")
    
    try:
        data = response.json()
        return data.get(response, "").strip()
    
    except json.JSONDecodeError:
        ## for somereason if the response comes in line-delimited.
        ## Ollama’s /api/generate endpoint doesn’t always return a single JSON object — it streams multiple JSON chunks by default, 
        # even if you set "stream": False

        full_result = ""
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            full_result += data.get("response", "")
        return full_result
