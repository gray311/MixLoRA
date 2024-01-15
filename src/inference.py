from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import torch
from safetensors import safe_open


# tensors = {}
# with safe_open("/home/myz/MixLoRA/ckpts/flan-t5-base/mmlu/mixlora/adapter_model.safetensors", framework="pt", device="cpu") as f:
#    for key in f.keys():
#        tensors[key] = f.get_tensor(key)
#        print(key, tensors[key].shape)


# Where peft_model_id should be the saving directory or huggingface model id
model_name = "google/flan-t5-base"
peft_model_id = "/home/myz/MixLoRA/ckpts/flan-t5-base/mmlu/mixlora"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
peft_model = PeftModel.from_pretrained(base_model, peft_model_id)

ckpt = torch.load("/home/myz/MixLoRA/ckpts/flan-t5-base/mmlu/mixlora/checkpoint-1000/pytorch_model.bin")
peft_model.load_state_dict(ckpt, strict=False)


# Input an instruction or any other questions.
prompt = """Which factor will most likely cause a person to develop a fever?
A. a leg muscle relaxing after exercise
B. a bacterial population in the bloodstream
C. several viral particles on the skin
D. carbohydrates being digested in the stomach
Answer: 
"""
inputs = tokenizer(prompt, return_tensors="pt")

outputs_peft = peft_model.generate(**inputs, max_length=2, do_sample=True)
outputs = base_model.generate(**inputs, max_length=2, do_sample=True)

print(tokenizer.batch_decode(outputs_peft, skip_special_tokens=True))
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))



