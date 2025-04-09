from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model and tokenizer
model_path = "/Users/pection/mike/emnlp/model/040925_062520"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Your input
sent0 = "He poured orange juice on his cereal."
sent1 = "He poured milk on his cereal."

# Format input based on your training setup (adapt as needed)
input_text = f"explain: {sent0} </s> {sent1}"

# Tokenize
inputs = tokenizer.encode(input_text, return_tensors="pt")

# Generate prediction
outputs = model.generate(inputs, max_length=64)

# Decode
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Model output:", decoded_output)
