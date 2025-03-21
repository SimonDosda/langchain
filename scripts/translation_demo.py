from langchain_community.llms import CTransformers
from langchain.schema import HumanMessage, SystemMessage, BaseMessage
from config import MODELS_DIR
from scripts.model_setup import setup_local_model

def translate_to_french(llm: CTransformers, text: str) -> str:
    messages: list[BaseMessage] = [
        SystemMessage(content="You are a professional English to French translator. Translate the given English text to French. Only provide the translation without any additional text or explanations."),
        HumanMessage(content=text)
    ]
    
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