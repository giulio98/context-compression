import wandb


def log_wandb_table(accelerator, predictions, ground_truth):
    table = wandb.Table(columns=["id", "question", "context", "prediction", "ground_truths"])
    for pred, ref in zip(predictions, ground_truth):
        table.add_data(pred['id'], ref['question'], ref["context"], pred['prediction_text'], ref['answers']["text"])
    accelerator.log({"Prediction Results": table})

def log_wandb_table_with_compression(accelerator, predictions, compressions, ground_truth):
    table = wandb.Table(columns=["id", "question", "context", "prediction", "compressed text", "ground_truths"])
    for pred, comp, ref in zip(predictions, compressions, ground_truth):
        table.add_data(pred['id'], ref['question'], ref["context"], pred['prediction_text'], comp['compressed_text'], ref['answers']["text"])
    accelerator.log({"Prediction Results": table})
