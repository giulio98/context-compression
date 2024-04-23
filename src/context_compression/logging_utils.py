import os

import wandb
import csv


def extract_ground_truth(example):
    answers = example['answers']
    # If 'answers' is a dictionary with a 'text' key, and the value is a list
    if isinstance(answers, dict) and 'text' in answers and isinstance(answers['text'], list):
        return ' | '.join(ans for ans in answers['text'])
    # If 'answers' is a list (assuming it's a list of strings)
    elif isinstance(answers, list):
        return ' | '.join(answers)
    else:
        # Add error handling for unexpected formats
        raise ValueError(f"Unexpected format of 'answers' in the dataset: {answers}")


def log_predictions_as_csv(predictions, references, file_path, data_config):
    fieldnames = ["id", "question", "prediction_text", "ground_truth_text"]

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for pred, ref in zip(predictions, references):
            ground_truth_text = extract_ground_truth(ref)
            row = {
                "id": pred['id'],
                "question": ref[data_config.question_column],
                "prediction_text": pred['prediction_text'],
                "ground_truth_text": ground_truth_text
            }
            writer.writerow(row)

def log_wandb_table(accelerator, predictions, ground_truth, data_config, num_rows=10):
    table = wandb.Table(columns=["id", "question", "prediction", "ground_truths"])
    for pred, ref in zip(predictions[:num_rows], ground_truth[:num_rows]):
        ground_truth_text = extract_ground_truth(ref)
        table.add_data(pred['id'], ref[data_config.question_column], pred['prediction_text'], ground_truth_text)
    accelerator.log({"Prediction Results": table})
