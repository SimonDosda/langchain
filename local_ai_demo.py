from langchain_community.llms import CTransformers
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from config import MODEL_CONFIG, MODELS_DIR

def setup_local_model():
    # Initialize the model with streaming output
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    # Define the model path
    model_path = os.path.join(MODELS_DIR, MODEL_CONFIG["name"])
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Please download the model and place it in the models directory."
        )
    
    # Load the model
    llm = CTransformers(
        model=model_path,
        model_type=MODEL_CONFIG["type"],
        config={
            'max_new_tokens': MODEL_CONFIG["max_new_tokens"],
            'temperature': MODEL_CONFIG["temperature"]
        },
        callback_manager=callback_manager
    )
    
    return llm

def main():
    print("Setting up local AI model...")
    llm = setup_local_model()
    
    # Test the model
    prompt = "What is the capital of France?"
    print("\nTesting the model with a simple question:")
    response = llm(prompt)
    print("\nModel response:", response)

if __name__ == "__main__":
    main() 