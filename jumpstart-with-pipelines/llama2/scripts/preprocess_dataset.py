import os
os.system('python3 -m pip install -U sagemaker datasets')

import json
import argparse
from datasets import load_dataset



def read_parameters():
    """
    Read job parameters
    Returns:
        (Namespace): read parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="databricks/databricks-dolly-15k")
    parser.add_argument('--category', type=str, default="summarization")
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--random_state', type=int, default=10)
    parser.add_argument('--input_path', type=str, default="/opt/ml/processing/input")
    parser.add_argument('--output_path_train', type=str, default="/opt/ml/processing/output/train")
    parser.add_argument('--output_path_validation', type=str, default="/opt/ml/processing/output/validation")
    parser.add_argument('--output_path_test', type=str, default="/opt/ml/processing/output/test")
    params, _ = parser.parse_known_args()
    return params


if __name__ == "__main__":
    
    # reading job parameters
    args = read_parameters()

    dolly_dataset = load_dataset(args.dataset_name, split="train")
    
    summarization_dataset = dolly_dataset.filter(lambda example: example["category"] == args.category)
    summarization_dataset = summarization_dataset.remove_columns("category")

    # Train + Validation and Test split
    train_and_test_dataset = summarization_dataset.train_test_split(
        test_size=args.test_size
    )
    train_valid_dataset, test_dataset = train_and_test_dataset["train"], train_and_test_dataset["test"] 
    
    # Train and Validation split
    train_and_valid_dataset = train_valid_dataset.train_test_split(
        test_size=args.val_size
    )
    train_dataset, valid_dataset = train_and_valid_dataset["train"], train_and_valid_dataset["test"]
    
    template = {
        "prompt": "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n",
        "completion": " {response}",
    }
    
    for _path in [
        args.output_path_train, 
        args.output_path_validation, 
        args.output_path_test
    ]:
        with open(os.path.join(_path, "template.json"), "w") as f:
            json.dump(template, f)

    # Dumping the training data to a local file to be used for training.
    train_dataset.to_json(os.path.join(args.output_path_train, "training.jsonl"))
    valid_dataset.to_json(os.path.join(args.output_path_validation, "validation.jsonl"))
    test_dataset.to_json(os.path.join(args.output_path_test, "test.jsonl"))
    
    print("Done")
        
        
