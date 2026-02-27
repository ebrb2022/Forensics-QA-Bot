# Forensics Q&A Bot

A bot that answers questions about forensics from relevent documents using Retrieval Augmented Generation (RAG). 

Model used: Meta Llama 3.2 3B Instruct

## Link to Streamlit App
<!-- Go [here](https://ebrb2022-ai-job-assistant-app-r3mair.streamlit.app/) to access the AI Job Assistant Streamlit app. -->
## Note
You can add more documents to the "sources" folder and run ingest.py to add them to database. I would recommend uncommenting the docs that have already been ingested to avoid rerunning them.

## Installation
1. Create a virtual environment and install the required packages via requirements.txt:
```
pip install -r requirements.txt
```
2. Create .env file and add your Hugging Face API key:
```HF_TOKEN = your_api_key
```

## Running the App
run the app.py file to start streamlit using the command: *streamlit run app.py*


