from langchain_community.llms import LlamaCpp
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from typing import List, Tuple
from config import MODELS_DIR
from scripts.model_setup import setup_local_model


def chat_with_ai(llm: LlamaCpp, user_input: str, history: List[Tuple[str, str]]) -> str:
    # Create the messages using the template and history
    messages = [
        SystemMessage(content="""You are a helpful and friendly AI assistant. You provide clear, concise, and accurate responses.
    You are knowledgeable about various topics and can engage in meaningful conversations.
    Always maintain a professional yet approachable tone.
    Use the conversation history to maintain context and provide relevant responses.
    Never continue or complete the user's sentences or thoughts.""")
    ]
    
    # Add conversation history
    for human_msg, ai_msg in history:
        messages.append(HumanMessage(content=f"User: {human_msg}"))
        messages.append(AIMessage(content=f"Assistant: {ai_msg}"))
    
    # Add current user input with explicit formatting
    messages.append(HumanMessage(content=f"User: {user_input}"))
    messages.append(HumanMessage(content="Assistant: "))
    
    return llm.invoke(messages)

def main() -> None:
    print("Setting up local AI model...")
    llm = setup_local_model()
    
    print("\nWelcome to the AI Chatbot!")
    print("Type 'quit' to exit the conversation.")
    print("Type 'clear' to start a new conversation.")
    print("Ask me anything!\n")
    
    # Initialize conversation history
    history: List[Tuple[str, str]] = []
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        # Check for commands
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nGoodbye! Have a great day!")
            break
        elif user_input.lower() == 'clear':
            history.clear()
            print("\nStarting a new conversation...\n")
            continue
        
        # Skip empty inputs
        if not user_input:
            continue
        
        # Get AI response
        try:
            print("\nAI: ", end="")
            response = chat_with_ai(llm, user_input, history)
            print(response)
            
            # Add to history
            history.append((user_input, response))
            
        except Exception as e:
            print(f"\nError: {str(e)}")
        
        print()  # Add a blank line between exchanges
