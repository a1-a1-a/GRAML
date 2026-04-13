
import argparse  
import json  
import os  

os.environ["UNSLOTH_RETURN_LOGITS"] = "1" 



try:
    from unsloth.models import _utils  
    def _no_op_get_statistics(*args, **kwargs):  
        return None  
    _utils._get_statistics = _no_op_get_statistics  
    print("Successfully patched Unsloth statistics check to allow offline loading.")  
except ImportError:  
    pass  


from typing import List, Dict, Any  
import torch  
from datasets import Dataset  
from unsloth import FastLanguageModel, is_bfloat16_supported  
from trl import SFTTrainer  
from transformers import TrainingArguments  


def read_json(path: str) -> List[Dict[str, Any]]: 
    
    with open(path, "rb") as f:  
        text = f.read().decode("utf-8-sig")  
        data = json.loads(text)  
    if isinstance(data, dict) and "data" in data: 
        data = data["data"]  
    return data  

def normalize_detect_label(v: Any) -> str: 

    s = str(v).strip() 
    if s in ["Yes", "No"]: 
        return s  
    if s.upper() in ["VULNERABLE", "YES", "1", "TRUE"]: 
        return "Yes"  
    if s.upper() in ["NOT VULNERABLE", "NO", "0", "FALSE"]:  
        return "No"  
    return "Yes" if s else "No"  

def build_prompt(system_prompt: str, instruction: str, user_input: str) -> str: 
    user = instruction.strip() 
    if user_input:  
        user = user + "\n## Input:\n" + str(user_input).strip() 
    return f"<s>[INST] <<SYS>>{system_prompt}<</SYS>>\n{user} [/INST]" 

def build_prompt_deepseek(system_prompt: str, instruction: str, user_input: str) -> str: 
    
    user = instruction.strip()  
    if user_input:  
        user = user + "\n" + str(user_input).strip() 
    
    return f"{system_prompt}\n### Instruction:\n{user}\n### Response:\n"  

def build_prompt_alpaca(system_prompt: str, instruction: str, user_input: str) -> str:  
    user = instruction.strip()
    input_str = ""
    if user_input:
        input_str = f"\n\n### Input:\n{str(user_input).strip()}"
    
    return f"{system_prompt}\n\n### Instruction:\n{user}{input_str}\n\n### Response:\n"

def build_prompt_qwen(system_prompt: str, instruction: str, user_input: str) -> str:
    user = instruction.strip()
    if user_input:
        user = user + "\n" + str(user_input).strip()
    
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

def prepare_dataset_entries(items: List[Dict[str, Any]], system_prompt: str, prompt_style: str = "mistral") -> List[Dict[str, str]]:
    res = []
    
    if prompt_style == "deepseek":
        build_fn = build_prompt_deepseek
    elif prompt_style == "alpaca":
        build_fn = build_prompt_alpaca
    elif prompt_style == "qwen":
        build_fn = build_prompt_qwen
    else:
        build_fn = build_prompt
        
    for it in items:
        instr = it.get("instruction", it.get("Instruction", ""))
        inp = it.get("input", it.get("Input", ""))
        out = it.get("output", it.get("Output", it.get("gold", it.get("detection", ""))))
        task = it.get("Task", it.get("task", it.get("category", it.get("type", "")))).strip().lower()
        
        prompt = build_fn(system_prompt, instr, inp)
        
        if task == "detection":
            label = normalize_detect_label(out)
        else:
            label = str(out).strip()
            
        if prompt_style == "qwen":
            label += "<|im_end|>"
            
        res.append({"text": prompt + " " + label, "prompt": prompt, "completion": " " + label})
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--train_path", type=str, default=r"D:\APP\VulLLM-main\VulLLM-main\gpt5\Ultimate2_train.json")
    parser.add_argument("--valid_path", type=str, default=r"D:\APP\VulLLM-main\VulLLM-main\gpt5\Ultimate_valid.json")
    parser.add_argument("--output_dir", type=str, default=r"D:\APP\VulLLM-main\VulLLM-main\wistral\mistral7b_unsloth")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=16384)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--system_prompt", type=str, default="You are a security assistant for vulnerability detection, assessment, location and description.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_style", type=str, default="mistral", choices=["mistral", "deepseek", "alpaca", "qwen"], help="Prompt template style")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory to save downloaded models")
    args = parser.parse_args()

    print(f"Loading Unsloth model: {args.model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_seq_length,
        dtype = None,
        load_in_4bit = True,
        fix_tokenizer = False,
        cache_dir = args.cache_dir,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = args.lora_alpha,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = args.seed,
        use_rslora = False,
        loftq_config = None,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    print("Loading and formatting datasets...")
    train_data_raw = read_json(args.train_path)
    valid_data_raw = read_json(args.valid_path)

    train_entries = prepare_dataset_entries(train_data_raw, args.system_prompt, args.prompt_style)
    valid_entries = prepare_dataset_entries(valid_data_raw, args.system_prompt, args.prompt_style)

    train_dataset = Dataset.from_list(train_entries)
    valid_dataset = Dataset.from_list(valid_entries)

    from transformers import DataCollatorForLanguageModeling
    import numpy as np

    class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
        def __init__(self, response_template, tokenizer, mlm=False, ignore_index=-100):
            super().__init__(tokenizer=tokenizer, mlm=mlm)
            self.response_template = response_template
            self.ignore_index = ignore_index
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)

        def torch_call(self, examples):
            batch = super().torch_call(examples)
            
            for i in range(len(examples)):
                input_ids = batch["input_ids"][i]
                
                start_idx = -1
                input_ids_list = input_ids.tolist()
                
                n = len(self.response_token_ids)
                for j in range(len(input_ids_list) - n + 1):
                    if input_ids_list[j : j + n] == self.response_token_ids:
                        start_idx = j
                        break
                
                if start_idx != -1:
                    end_of_prompt_idx = start_idx + n
                    batch["labels"][i, :end_of_prompt_idx] = self.ignore_index
                else:
                    batch["labels"][i, :] = self.ignore_index

            return batch

    if args.prompt_style == "deepseek":
        response_template = "### Response:\n"
    elif args.prompt_style == "alpaca":
        response_template = "### Response:\n"
    elif args.prompt_style == "qwen":
        response_template = "<|im_start|>assistant\n"
    else:
        response_template = "[/INST]"
        
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer
    )

    training_args = TrainingArguments(
        output_dir = args.output_dir,
        per_device_train_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        warmup_ratio = 0.05,
        num_train_epochs = args.num_train_epochs,
        learning_rate = args.learning_rate,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = args.seed,
        save_strategy = "epoch",
        eval_strategy = "epoch",
        report_to = "none",
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = valid_dataset,
        dataset_text_field = "text",
        max_seq_length = args.max_seq_length,
        data_collator = collator,
        args = training_args,
        packing = False,
    )

    print("Starting training...")
    trainer_stats = trainer.train()

    print(f"Saving model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    

    print("Training finished!")

if __name__ == "__main__":
    main()
