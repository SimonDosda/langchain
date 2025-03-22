from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from config import MODEL_CONFIG, MODELS_DIR

def setup_local_model() -> LlamaCpp:
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
    
    # Load the model with adjusted parameters
    llm = LlamaCpp(
        model_path=model_path,
        temperature=MODEL_CONFIG["temperature"],
        max_tokens=MODEL_CONFIG["max_new_tokens"],
        callback_manager=callback_manager,
        verbose=True,
        stop=["User:", "Human:", "Assistant:", "AI:", "\n\n"],  # Stop tokens to prevent continuing user input
        echo=False,  # Don't echo the input
        top_p=0.95,  # Nucleus sampling
        repeat_penalty=1.1,  # Penalty for repeating tokens
        n_ctx=2048,  # Context window
        n_threads=4,  # Number of CPU threads to use
        f16=True,  # Use half precision for better performance
        n_batch=512  # Batch size for prompt processing
    )
    
    return llm 