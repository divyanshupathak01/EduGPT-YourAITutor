import os
from google import genai

with open(".env", "r") as f:
    env_file = f.readlines()
envs_dict = {
    key.strip("'"): value.strip("\n")
    for key, value in [i.split("=", 1) for i in env_file if "=" in i]
}
api_key = envs_dict.get("GOOGLE_API_KEY", envs_dict.get("OPENAI_API_KEY", ""))

try:
    client = genai.Client(api_key=api_key)
    print("Available Gemini Models:")
    print("-" * 20)
    for model in client.models.list():
        if "gemini" in model.name.lower():
            print(model.name)
except Exception as e:
    print("Error:", e)
