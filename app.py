from flask import Flask, request, render_template, jsonify
from flask_wtf.csrf import CSRFProtect
import pandas as pd
from openai import OpenAI
import os
import logging
from dotenv import load_dotenv
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET', 'dev-secret-123')
csrf = CSRFProtect(app)

# Validate OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable missing")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Load your extended CSV file
data = pd.read_csv('students.csv')

# Convert the full CSV data to a string (without the index)
full_data_str = data.to_csv(index=False)

def ask_question_gpt4(question: str, full_data: str) -> str:
    """Analyze student data using OpenAI's API with improved safety and error handling."""
    try:
        if not question.strip():
            return "❌ Please enter a valid question"
            
        if len(question) > 200:
            return "❌ Question too long (max 200 characters)"

        prompt = f"""CSV data:\n{full_data}\n
Q: {question[:200]}  # Truncate question to 200 chars
Rules:
1. Answer in EXACTLY 4 lines
2. Use ONLY this format:
🎯 ANSWER: Single clear statement
📊 DATA: Maximum 2 key numbers
✨ TOP: List max 3 items
💡 KEY: One insight in 5-7 words"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a school data analyst. Respond only with factual insights from the data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.3
        )
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return "❌ Analysis service unavailable. Please try again later."

def prepare_dashboard_stats(df) -> Dict[str, Any]:
    """Generate dashboard statistics with data validation and error handling."""
    try:
        required_columns = {'Name', 'Class', 'Marks_Math', 'Marks_Science', 'Marks_English'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"CSV missing required columns: {', '.join(missing)}")

        return {
            'total_students': len(df),
            'class_distribution': df['Class'].value_counts().to_dict(),
            'subject_averages': {
                'Math': round(float(df['Marks_Math'].mean()), 2),
                'Science': round(float(df['Marks_Science'].mean()), 2),
                'English': round(float(df['Marks_English'].mean()), 2)
            },
            'top_performers': df.nlargest(
                5, 
                ['Marks_Math', 'Marks_Science', 'Marks_English']
            ).drop_duplicates()[
                ['Name', 'Class', 'Marks_Math', 'Marks_Science', 'Marks_English']
            ].to_dict('records'),
            'class_performance': df.groupby('Class')[
                ['Marks_Math', 'Marks_Science', 'Marks_English']
            ].mean().round(2).to_dict()
        }
    except Exception as e:
        logger.error(f"Data processing error: {str(e)}")
        return {
            'error': f"Data processing error: {str(e)}"
        }

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
