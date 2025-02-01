import os
import logging
from dotenv import load_dotenv
from typing import Dict, Any
from flask import Flask, request, render_template
from flask_wtf.csrf import CSRFProtect
from flask_wtf import FlaskForm
from wtforms import StringField
import pandas as pd
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET', 'dev-secret-123')
csrf = CSRFProtect(app)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load your extended CSV file
data = pd.read_csv('students.csv')
logger.info("CSV data loaded successfully.")

# Convert the full CSV data to a string (without the index)
full_data_str = data.to_csv(index=False)

@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(3))
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
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=150,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in ask_question_gpt4: {str(e)}")
        return "❌ Analysis service unavailable. Please try again later."

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

class QuestionForm(FlaskForm):
    question = StringField('Question')

@app.route('/', methods=['GET', 'POST'])
def index():
    form = QuestionForm()
    answer = None
    question = ""
    if form.validate_on_submit():
        question = form.question.data
        if question:
            answer = ask_question_gpt4(question, full_data_str)
            answer = answer.replace('\n', '<br>')
    
    stats = prepare_dashboard_stats(data)
    data_summary = {
        'total_rows': len(data),
        'columns': list(data.columns)
    }
    
    return render_template(
        'index.html', 
        form=form,
        stats=stats,
        data_summary=data_summary,
        question=question, 
        answer=answer
    )

if __name__ == '__main__':
    app.run(debug=True)