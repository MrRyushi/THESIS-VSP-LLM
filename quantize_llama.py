from transformers import AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM

# Load the original Llama-2 model
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)

# Apply GPTQ quantization
quantized_model = AutoGPTQForCausalLM.from_pretrained(model, quantize_config="auto")

# Save the quantized model
quantized_model.save_pretrained("Llama-2-7b-GPTQ")

print("✅ Quantization complete! Model saved as 'Llama-2-7b-GPTQ'.")
