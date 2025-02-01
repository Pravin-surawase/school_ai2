from flask import Flask, request, render_template
import pandas as pd
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Set your OpenAI API key from environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Load your extended CSV file
data = pd.read_csv('students.csv')

# Convert the full CSV data to a string (without the index)
full_data_str = data.to_csv(index=False)

def ask_question_gpt4(question, full_data):
    try:
        prompt = (
            f"CSV data:\n{full_data}\n"
            f"Q: {question}\n"
            f"Rules:\n"
            f"1. Answer in EXACTLY 4 lines\n"
            f"2. Use ONLY this format:\n"
            f"🎯 ANSWER: Single clear statement\n"
            f"📊 DATA: Maximum 2 key numbers\n"
            f"✨ TOP: List max 3 items\n"
            f"💡 KEY: One insight in 5-7 words\n"
        )
        
        messages = [
            {
                "role": "system", 
                "content": "You are a minimal data reporter. Never explain. Never add context. Just facts."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=150,
            temperature=0.3
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"❌ Error: {str(e)}"

def prepare_dashboard_stats(df):
    stats = {
        'total_students': len(df),
        'class_distribution': df['Class'].value_counts().to_dict(),
        'subject_averages': {
            'Math': df['Marks_Math'].mean().round(2),
            'Science': df['Marks_Science'].mean().round(2),
            'English': df['Marks_English'].mean().round(2)
        },
        'top_performers': df.nlargest(5, ['Marks_Math', 'Marks_Science', 'Marks_English'])[
            ['Name', 'Class', 'Marks_Math', 'Marks_Science', 'Marks_English']
        ].to_dict('records'),
        'class_performance': df.groupby('Class')[
            ['Marks_Math', 'Marks_Science', 'Marks_English']
        ].mean().round(2).to_dict()
    }
    return stats

@app.route('/', methods=['GET', 'POST'])
def index():
    # Basic summary statistics
    data_summary = {
        'total_rows': len(data),
        'columns': list(data.columns)
    }
    
    stats = prepare_dashboard_stats(data)
    answer = None
    question = ""
    if request.method == 'POST':
        question = request.form.get('question')
        if question:
            answer = ask_question_gpt4(question, full_data_str)
            answer = answer.replace('\n', '<br>')
            
    return render_template('index.html', 
                         stats=stats,
                         data_summary=data_summary,
                         question=question, 
                         answer=answer)

if __name__ == '__main__':
    app.run(debug=True)