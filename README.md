# LangChain Local AI Demo

This project demonstrates how to use LangChain with a local AI model using the ctransformers library.

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
├── config.py        # Configuration settings
├── local_ai_demo.py # Main script
├── requirements.txt # Project dependencies
└── README.md       # This file
```

## Usage

Run the demo script:
```bash
python local_ai_demo.py
```

## Notes

- The script uses Llama 2 as an example, but you can use other models supported by ctransformers
- Make sure you have enough RAM available (at least 8GB recommended)
- The first run might take a few minutes as the model loads into memory
- Model files are stored in the `models` directory and are not tracked in git
- You can modify model settings in `config.py` without changing the main code 