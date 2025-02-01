from flask import Flask, request, render_template
import pandas as pd
import openai
import os

app = Flask(__name__)

# Set your OpenAI API key from environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Load your extended CSV file
data = pd.read_csv('students.csv')

# Convert the full CSV data to a string (without the index)
full_data_str = data.to_csv(index=False)

def ask_question_gpt4(question, full_data):
    # Build a prompt that includes the full CSV data
    prompt = (
        f"Below is the full data from a school's CSV file:\n\n"
        f"{full_data}\n\n"
        f"Based on this data, please answer the following question:\n"
        f"{question}"
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided school CSV data."},
        {"role": "user", "content": prompt}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # Using GPT-4 model
        messages=messages,
        max_tokens=200,
        temperature=0.5
    )
    return response.choices[0].message['content']

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    question = ""
    if request.method == 'POST':
        question = request.form.get('question')
        if question:
            answer = ask_question_gpt4(question, full_data_str)
    # Optionally, you could also show a truncated version of the data on the page.
    return render_template('index.html', question=question, answer=answer, full_data=full_data_str)

if __name__ == '__main__':
    app.run(debug=True)