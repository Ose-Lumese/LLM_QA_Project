import os
from flask import Flask, render_template, request

app = Flask(__name__)

from google import genai
from google.genai import types
import nltk
import string

# Download NLTK resources to a temporary location if running remotely
# The current directory is typically writable in PaaS environments.
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    # Set the NLTK data directory to a location writable by the application
    nltk.download('punkt', quiet=True)


# 1. Initialize the Gemini Client
# Client will automatically pick up GEMINI_API_KEY from environment variables on Render
try:
    client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    client = None

def preprocess_question(question):
    """Applies basic preprocessing (lowercasing, tokenization, punctuation removal)."""
    text = question.lower()
    tokens = nltk.word_tokenize(text)
    # Remove punctuation from tokens
    tokens = [word.translate(str.maketrans('', '', string.punctuation)) for word in tokens]
    processed_question = ' '.join(token for token in tokens if token)
    return processed_question

def get_llm_answer(prompt):
    """Constructs a prompt and sends it to the Gemini API."""
    if not client:
        return "ERROR: LLM client is not initialized. Check server logs."
        
    system_instruction = "You are a helpful and concise Question-and-Answering system. Provide a direct and accurate answer to the user's question."
    
    config = types.GenerateContentConfig(
        system_instruction=system_instruction
    )
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=config
        )
        return response.text
    except Exception as e:
        return f"An error occurred with the LLM API: {e}"

@app.route('/', methods=['GET', 'POST'])
def index():
    question = None
    processed_question = None
    llm_response = None
    
    if request.method == 'POST':
        question = request.form.get('question', '')
        
        if question:
            # 1. View the processed question
            # NOTE: We send the original question to the LLM for the best result, 
            # but preprocess it here for display/demonstration purposes.
            processed_question = preprocess_question(question)
            
            # 2. See the LLM API response / 3. Display the generated answer
            llm_response = get_llm_answer(question)
            
    # Pass all variables to the HTML template
    return render_template('index.html', 
                            question=question, 
                            processed_question=processed_question, 
                            llm_response=llm_response)

if __name__ == '__main__':
    # When running locally, Flask runs directly
    # For Render, Gunicorn runs the app using 'gunicorn app:app'
    app.run(debug=True)