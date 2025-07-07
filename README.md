# mini RAG App

This is a toy project to demonstrate how to use RAG (Retrieval-Augmented Generation) to build a question-answering chatbot with a customized literature database.

This app is based on LLMs, so you need to either have access to a LLM API or deploy a LLM locally (check `llm_api.py` for more details).

## Setup

This project was developed with Python 3.10. Any version of Python 3.8 or above should work.

### 1. Clone the repository

```bash
git clone https://github.com/jasonzhang-zzx/mini_rag_app.git
cd mini_rag_app
```
### 2. Create a new environment and install the required packages
If you are using Anaconda:

```bash
conda env create -f environment.yml
conda activate rag_test
```
If you are using pip and venv:

```bash
python -m venv rag_test
source rag_test/bin/activate
pip install -r requirements.txt
```
### 3. Set up the LLM API
A local deployment of the LLM is recommended. If you have never deployed a LLM locally before, here is a helpful  [tutorial](https://www.machinelearningplus.com/gen-ai/ollama-tutorial-your-guide-to-running-llms-locally/).

If you are using a LLM API, you need to set up the API key in a .env file. The code for handling this is already prepared — check `llm_api.py` for more details.

### 4. Run the app
Copy all your literature files to the `literatures/` folder.

Run `vectorDB.py` to initialize the vector database:

```bash
python vectorDB.py
```
Then run `main.py` to start the chatbot:

```bash
python main.py
```


## License
MIT © 2025 Jason Zhang