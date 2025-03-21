# LangChain Local AI Demo

This project demonstrates how to use LangChain with a local AI model using the ctransformers library. It provides a simple English to French translation service.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download a model file:
   - Visit [TheBloke's Hugging Face page](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)
   - Download a GGUF model file (e.g., `llama-2-7b-chat.gguf`)
   - Place it in the `models` directory

4. Configure the model:
   - Edit `config.py` to set your model name and parameters
   - The default configuration uses Llama 2 7B Chat model

## Project Structure

```
.
├── models/           # Directory for model files
├── scripts/         # Demo scripts
│   ├── __init__.py
│   ├── model_setup.py
│   └── translation_demo.py
├── config.py        # Configuration settings
├── run.py          # Script runner
├── requirements.txt # Project dependencies
└── README.md       # This file
```

## Usage

Run the demos using the script runner:

```bash
# Show available demos and usage
python run.py

# Run the translation demo
python run.py translation
# or
./run.py translation
```

The program will:
1. Load the local AI model
2. Present an interactive prompt
3. Accept English text input
4. Translate the input to French
5. Display the translation

To exit the program, type 'quit' when prompted for input.

## Notes

- The script uses Llama 2 as an example, but you can use other models supported by ctransformers
- Make sure you have enough RAM available (at least 8GB recommended)
- The first run might take a few minutes as the model loads into memory
- Model files are stored in the `models` directory and are not tracked in git
- You can modify model settings in `config.py` without changing the main code 