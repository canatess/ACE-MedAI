import pandas as pd
import openai
from tqdm import tqdm

# --- SETUP ---

# Correct way to initialize the OpenAI client in v1.0+
client = openai.OpenAI(api_key="OPENAI_API_KEY")

# Load the full CSV
csv_path = r"csvfile.csv"  # Replace with your actual CSV file path
df = pd.read_csv(csv_path)

# Only first 5 rows
df_sample = df.head(5).copy()

# --- PROMPT TEMPLATE ---

prompt_template = """
You are a radiology assistant LLM. Given a radiology question-answering instance with the associated report and options, generate a reasoning trace that mimics a radiologist's clinical reasoning.

Format your output in this structure:
<think> your reasoning here step-by-step </think>
<answer> your final answer here (e.g., A, B, C, or D) </answer>

Here is the data:
Question: {question}
Options:
{options}
Findings: {findings}
Impression: {impression}
Indication: {indication}

Now, reason through the problem.
"""

# --- GPT CALL FUNCTION ---

def call_gpt4o(question, options, findings, impression, indication):
    prompt = prompt_template.format(
        question=question or "",
        options=options or "",
        findings=findings or "",
        impression=impression or "",
        indication=indication or "",
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {str(e)}"

# --- PROCESS ROWS ---

outputs = []
for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    result = call_gpt4o(
        question=row.get("question"),
        options=row.get("options"),
        findings=row.get("Findings_report"),
        impression=row.get("Impression_report"),
        indication=row.get("Indication_report"),
    )
    outputs.append(result)

# Save output
df_sample["gpt4o_reasoning_output"] = outputs
df_sample.to_csv("radiology_gpt4o_outputs_full_columns.csv", index=False)

print("✅ Done! Output saved to radiology_gpt4o_outputs_full_columns.csv")
