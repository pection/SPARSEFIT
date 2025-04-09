import gradio as gr
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "/Users/pection/mike/emnlp/model/base_line_cos_e/040925_072112"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
model.eval()

# Inference function
def generate_text(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    decoder_input_ids = torch.full(
        (1, 1), tokenizer.pad_token_id, dtype=torch.long, device=device
    )
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=300,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id=tokenizer.pad_token_id
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

# Gradio UI
demo = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(lines=3, placeholder="Enter your input text for T5-base..."),
    outputs="text",
    title="T5-base Text Generator",
    description="Try giving instructions like 'Translate English to German: Hello, how are you?'"
)

# Launch
demo.launch()
