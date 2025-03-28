from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.schema import SystemMessage, HumanMessage
from typing import List, Tuple
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from config import MODELS_DIR, MODEL_CONFIG
import json
from datetime import datetime

def setup_agent_model() -> LlamaCpp:
    """Set up a custom model for the agent without stop tokens."""
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
    
    # Load the model with adjusted parameters, but WITHOUT stop tokens
    llm = LlamaCpp(
        model_path=model_path,
        temperature=MODEL_CONFIG["temperature"],
        max_tokens=MODEL_CONFIG["max_new_tokens"],
        callback_manager=callback_manager,
        verbose=False,  # Disable verbose output to remove performance metrics
        # No stop tokens for agent usage
        echo=False,
        top_p=0.95,
        repeat_penalty=1.1,
        n_ctx=2048,
        n_threads=4,
        f16=True,
        n_batch=512
    )
    
    return llm

def create_calculator_tool() -> Tool:
    """Create a calculator tool for basic arithmetic operations."""
    def calculator(query: str) -> str:
        try:
            # Clean up the input
            query = query.strip()
            
            # Handle case where the query might be the expression directly
            if query.startswith("'") and query.endswith("'"):
                query = query[1:-1]  # Remove quotes
            
            # If query is empty or just whitespace
            if not query or query.isspace():
                return "I need a mathematical expression to calculate."
                
            # Extract numbers and operators, ignoring other text
            import re
            
            # remove any text after line break
            query = query.split('\n')[0]
            
            # Extract the mathematical expression
            expression = re.sub(r'[^0-9+\-*/().\s]', '', query)
            expression = expression.strip()
            
            if not expression:
                return "Could not identify a valid mathematical expression. Please provide a calculation like '2 + 2'."
            
            # Evaluate the expression using a safer approach
            import ast
            import operator
            
            # Define the allowed operators
            operators = {ast.Add: operator.add, ast.Sub: operator.sub,
                        ast.Mult: operator.mul, ast.Div: operator.truediv,
                        ast.USub: operator.neg}
            
            def eval_expr(node):
                if isinstance(node, ast.Num):
                    return node.n
                elif isinstance(node, ast.BinOp):
                    return operators[type(node.op)](eval_expr(node.left), eval_expr(node.right))
                elif isinstance(node, ast.UnaryOp):
                    return operators[type(node.op)](eval_expr(node.operand))
                else:
                    raise TypeError(f"Unsupported type: {node}")
            
            # Simple expressions like "2 + 4" can be safely evaluated with eval
            try:
                result = eval(expression)
                return f"{expression} = {result}"
            except:
                # For more complex expressions or if eval fails, fallback to a direct calculation
                if "+" in expression:
                    parts = expression.split("+")
                    try:
                        result = sum(float(p.strip()) for p in parts)
                        return f"{expression} = {result}"
                    except:
                        pass
                
                if "-" in expression and not expression.startswith("-"):
                    parts = expression.split("-")
                    try:
                        result = float(parts[0].strip())
                        for p in parts[1:]:
                            result -= float(p.strip())
                        return f"{expression} = {result}"
                    except:
                        pass
                
                if "*" in expression:
                    parts = expression.split("*")
                    try:
                        result = 1
                        for p in parts:
                            result *= float(p.strip())
                        return f"{expression} = {result}"
                    except:
                        pass
                
                if "/" in expression:
                    parts = expression.split("/")
                    try:
                        result = float(parts[0].strip())
                        for p in parts[1:]:
                            divisor = float(p.strip())
                            if divisor == 0:
                                return "Error: Cannot divide by zero"
                            result /= divisor
                        return f"{expression} = {result}"
                    except:
                        pass
                
                return "Could not evaluate the expression. Please provide a simpler calculation."
                
        except Exception as e:
            return f"Error calculating: {str(e)}. Please provide a simple expression like '2 + 4'."
    
    return Tool(
        name="Calculator",
        func=calculator,
        description="Useful for performing arithmetic calculations. Use this when you need to add, subtract, multiply, or divide numbers. Input should be a mathematical expression like '2 + 4' or '15 * 3'.",
    )

def create_search_tool() -> Tool:
    """Create a search tool for looking up information."""
    def search(query: str) -> str:
        # Clean up the input
        query = query.strip()
        
        # Handle quoted queries
        if query.startswith("'") and query.endswith("'"):
            query = query[1:-1]  # Remove quotes
            
        # Check if the query is the placeholder text
        if query.lower() == "your search query here" or query.lower() == "your search query here." or query.lower() == "your actual search query goes here":
            return "Please provide an actual search query instead of the placeholder text. For example, 'is the earth flat' or 'who invented the internet'."
        
        # If query is empty
        if not query or query.isspace():
            return "I need a search query to look up information."
        
        # This is a placeholder implementation
        # In a real application, you would integrate with a search API
        return f"Search results for '{query}': This is a simulated search result. In a real implementation, this would connect to a search API to provide actual results. For your query about '{query}', you would typically find relevant information online."
    
    return Tool(
        name="Search",
        func=search,
        description="Useful for finding information on the internet. Use this when you need to answer questions about current events, facts, or any information you're not certain about. Input should be a search query."
    )

def create_agent(llm) -> initialize_agent:
    """Initialize the agent with tools."""
    tools = [
        create_calculator_tool(),
        create_search_tool()
    ]
    
    # Initialize the agent with a more explicit format
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Using a simpler agent type that works better
        verbose=True,  # Show verbose output
        max_iterations=5,
        early_stopping_method="generate",
        handle_parsing_errors=True,
        agent_kwargs={
            "prefix": """You are a helpful AI assistant that can use tools to answer questions.
When you receive a question, think about what tools would help you answer it.
Always think step by step about what you need to do to accomplish the task.

IMPORTANT: For each tool, you must:
1. First state what tool you will use and why
2. Then call the tool with the EXACT proper format shown below:
   Action: Calculator
   Action Input: 2 + 4

   or

   Action: Search
   Action Input: your actual search query goes here, not this placeholder text

3. Wait for the observation (result) from the tool
4. Use that observation to formulate your final answer

For the Search tool, REPLACE 'your actual search query goes here' with the specific information you want to search for.
For example, if the user asks "search if the earth is flat", your Action Input should be "is the earth flat".

You have access to the following tools:""",
            "suffix": """Begin! Remember to always show your reasoning and explain what tools you're using and why.

Question: {input}
Thought: I need to carefully analyze this question and determine which tools to use.
{agent_scratchpad}"""
        }
    )
    
    return agent

def log_interaction(run_file: str, user_input: str, agent_output: str, intermediate_steps=None) -> None:
    """Log the interaction to a JSON file."""
    # Create runs directory if it doesn't exist
    os.makedirs("runs", exist_ok=True)
    
    # Load existing interactions or create new list
    interactions = []
    if os.path.exists(run_file):
        with open(run_file, 'r') as f:
            interactions = json.load(f)
    
    # Format intermediate steps if they exist
    formatted_steps = []
    if intermediate_steps:
        for step in intermediate_steps:
            if isinstance(step, tuple) and len(step) >= 2:
                action = step[0]
                observation = step[1]
                formatted_steps.append({
                    "action": f"{action.tool} - {action.tool_input}",
                    "observation": observation
                })
    
    # Add new interaction
    interactions.append({
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "agent_output": agent_output,
        "thought_process": formatted_steps if formatted_steps else []
    })
    
    # Save updated interactions
    with open(run_file, 'w') as f:
        json.dump(interactions, f, indent=2)

def main() -> None:
    print("Setting up local AI model...")
    llm = setup_agent_model()  # Use our custom model without stop tokens
    
    print("\nInitializing agent...")
    agent = create_agent(llm)
    
    print("\nWelcome to the AI Agent Demo!")
    print("This agent can perform calculations and search for information.")
    print("Type 'quit' to exit.")
    print("Ask me to perform any task!\n")
    
    # Initialize conversation history
    history: List[Tuple[str, str]] = []
    
    # Create run file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_file = f"runs/agent_run_{timestamp}.json"
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        # Check for quit command
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nGoodbye! Have a great day!")
            break
        
        # Skip empty inputs
        if not user_input:
            continue
        
        try:
            # Format user input with any previous context needed
            if history:
                # Add a brief summary of the conversation history to provide context
                last_exchanges = history[-2:] if len(history) > 2 else history
                context = "Based on our previous conversation:\n"
                for i, (q, a) in enumerate(last_exchanges):
                    context += f"- You asked: {q}\n- I answered: {a}\n"
                enhanced_input = f"{context}\nNow you're asking: {user_input}"
            else:
                enhanced_input = user_input
            
            # Get agent response
            response = agent.invoke({
                "input": enhanced_input
            })
            
            # Get intermediate steps if available
            intermediate_steps = response.get("intermediate_steps", [])
            
            # Display the intermediate steps (thought process)
            if intermediate_steps:
                print("\nThought process:")
                for step in intermediate_steps:
                    action = step[0]
                    observation = step[1]
                    print(f"Action: {action.tool} - {action.tool_input}")
                    print(f"Observation: {observation}\n")
            
            # Extract the output from the response
            if isinstance(response, dict):
                agent_output = response.get('output', str(response))
            else:
                agent_output = str(response)
            
            print("\nAgent:", agent_output)
            
            # Log the interaction
            log_interaction(run_file, user_input, agent_output, intermediate_steps)
            
            # Add to history
            history.append((user_input, agent_output))
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            # Log the error
            log_interaction(run_file, user_input, f"Error: {str(e)}", [])
        
        print()  # Add a blank line between exchanges

if __name__ == "__main__":
    main() 