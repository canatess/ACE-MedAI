import pandas as pd
import ollama
import json
import ast

# ==== SETTINGS ====
input_file = 'your_file.csv'
output_file = 'results_cot_2.csv'
model_name = 'deepseek-r1:70b'
start_row = 150
end_row = 200

# ==== READ CSV ====
df = pd.read_csv(input_file)

# ==== Define Target Modalities ====
target_modalities = {"Magnetic Resonance Imaging", "Computed Tomography"}

# ==== Output Collector ====
output_rows = []

# ==== FOR LOOP OVER A SLICE OF THE CSV ====
for idx in range(start_row, end_row):
    row = df.iloc[idx]
    modality = row["modality"]
    im_name = row["image"]

    if modality in target_modalities:
        # Try parsing the JSON in 'conversations' column
        try:
            raw_json = ast.literal_eval(row['conversations'])
            question = raw_json[0]['value']
            answer = raw_json[1]['value']
            
        except Exception as e:
            print("bbbbbbbbbbbb")

        try:
            print(f"🧠 Row {idx} | Modality: {modality}")
            print(answer)
            response = ollama.chat(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": f"""You are an *AI clinical assistant* with *advanced visual perception abilities, trained to interpret **medical images* and make *evidence-based clinical inferences. When given a visual description, you must interpret it **as if you are directly viewing the medical image. Your role is to demonstrate **structured clinical reasoning* grounded in both *visual understanding* and *medical expertise*.

---

## Instructions:
- Assume the image description represents what you see directly with your own eyes.
- Do *not* mention or imply that you are reading a caption or text.
- Only use the information contained in the visual description.
- Never hallucinate findings or infer beyond what is visually described.
- Interpret all spatial, numerical, or anatomical details confidently and naturally.
- Express detailed reasoning using <think> tags.
- Present your final conclusion or diagnosis using <answer> tags.
- Use a formal, medically accurate tone throughout.

---

## Output Format:

```text
<think>
[Step-by-step visual interpretation and clinical reasoning]
</think>
<answer>
[Final diagnosis, interpretation, or conclusion]
</answer>

Image Description:

{answer}

Question:

{question}"""
                    }
                ]
            )
            cot_output = response['message']['content'].strip()
            output_rows.append({
                    "filename": im_name,
                    "row_index": idx,
                    "modality": modality,
                    "cot_output": cot_output
                })

            

        except Exception as e:
            print("Error during Ollama chat:", e)

# ==== SAVE TO CSV ====
out_df = pd.DataFrame(output_rows)
out_df.to_csv(output_file, index=False)
print(f"✅ Saved {len(output_rows)} rows to: {output_file}")