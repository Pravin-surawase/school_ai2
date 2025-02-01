import openai
import pandas as pd

# Load your CSV file (adjust the file name as needed)
data = pd.read_csv('students.csv')

# Create a simple summary of your data (you can customize this as needed)
def summarize_data(df):
    summary = f"Total Students: {len(df)}\n"
    summary += f"Average Math Marks: {df['Marks_Math'].mean():.1f}\n"
    summary += f"Average Science Marks: {df['Marks_Science'].mean():.1f}\n"
    summary += f"Average English Marks: {df['Marks_English'].mean():.1f}\n"
    summary += f"Average Attendance: {df['Attendance'].mean():.1f}%"
    return summary

data_summary = summarize_data(data)
print("Data Summary:\n", data_summary)

# Set your OpenAI API key (replace with your actual API key)
import os  # Add this at the top if not already present

openai.api_key = os.environ.get("OPENAI_API_KEY")

def ask_question_gpt4(question, summary):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided school CSV data."},
        {"role": "user", "content": f"Data Summary:\n{summary}\n\nQuestion: {question}"}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # Specify GPT-4 as the model
        messages=messages,
        max_tokens=150,
        temperature=0.5
    )
    
    return response.choices[0].message.content.strip()

# Example query
question = "Which student has the highest Math score?"
answer = ask_question_gpt4(question, data_summary)
print("\nGPT-4 Answer:", answer)