from langchain_community.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from typing import List
from config import MODELS_DIR
from scripts.model_setup import setup_local_model

# Define the chat prompt template
TRANSLATION_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "You are a professional English to French translator. Translate the given English text to French. Only provide the translation without any additional text or explanations."),
    ("user", "{text}")
])

def translate_to_french(llm: LlamaCpp, text: str) -> str:
    # Create the messages using the template
    messages = TRANSLATION_TEMPLATE.format_messages(text=text)
    return llm.invoke(messages)

def main() -> None:
    print("Setting up local AI model...")
    llm = setup_local_model()
    
    print("\nWelcome to the French Translation Assistant!")
    print("Enter your message to translate to French:")
    
    # Get user input (one line only)
    user_input = input("> ").strip()
    
    # Translate the input
    print("\nTranslating to French...")
    try:
        french_translation = translate_to_french(llm, user_input)
        print(f"\nFrench translation: {french_translation}")
    except Exception as e:
        print(f"\nError: {str(e)}")