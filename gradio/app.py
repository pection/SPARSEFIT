from transformers import T5ForConditionalGeneration, T5Tokenizer
import gradio as gr

# Load your trained model and tokenizer
model_path = "/Users/pection/mike/emnlp/model/040925_062520"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

def predict(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=100)
    print(output)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output

gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=2, placeholder="Enter input..."),
    outputs="text",
    title="T5 Text Generation",
    description="Test your fine-tuned T5 model on custom inputs."
).launch()
