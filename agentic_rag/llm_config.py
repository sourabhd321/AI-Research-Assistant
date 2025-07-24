from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from typing import cast

# Load environment variables from .env file
load_dotenv()

# Access the key and model name
google_api_key = os.getenv("GOOGLE_API_KEY")
google_model_name = os.getenv("GOOGLE_MODEL_NAME")

print("API KEY:", google_api_key)
print("MODEL NAME:", google_model_name)


# Set up the LLM with the Google API key and model name
llm = ChatGoogleGenerativeAI(
    model=google_model_name,
    api_key=google_api_key,
    convert_system_message_to_human=True
)
