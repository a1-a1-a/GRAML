import json
import argparse
import os
import sys
import random
import transformations

def read_json(path):
    with open(path, "rb") as f:
        text = f.read().decode("utf-8-sig")
        data = json.loads(text)
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data

def apply_transforms(item, transform_names):
    code = item.get("input", "")
    if not code:
        return item
    
    new_code = code
    for tf_name in transform_names:
        if hasattr(transformations, tf_name):
            tf_func = getattr(transformations, tf_name)
            try:
                if tf_name in ["tf_10", "tf_11", "tf_13"]:
                    pass 
                else:
                    new_code = tf_func(new_code)
            except Exception as e:
                pass
        else:
            print(f"Warning: Transformation {tf_name} not found in transformations.py")
            
    new_item = item.copy()
    new_item["input"] = new_code
    return new_item

def main():
    parser = argparse.ArgumentParser(description="Apply C/C++ code perturbations using transformations.py")
    parser.add_argument("--input_path", type=str, required=True, help="Original test set JSON path")
    parser.add_argument("--output_path", type=str, required=True, help="Output path")
    parser.add_argument("--transforms", type=str, nargs='+', required=True,
                        help="Transformation list, e.g. tf_1 tf_6 (multiple supported)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print(f"Reading data from {args.input_path}...")
    try:
        data = read_json(args.input_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print(f"Applying transformations: {args.transforms}...")
    new_data = []
    
    from tqdm import tqdm
    for item in tqdm(data, desc="Processing"):
        new_data.append(apply_transforms(item, args.transforms))
        
    output_json = {"data": new_data}
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    
    print(f"Saving to {args.output_path}...")
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)
        
    print("Done!")

if __name__ == "__main__":
    main()
