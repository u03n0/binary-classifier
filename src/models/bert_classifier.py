import torch
import pandas as pd

from typing import Tuple, List
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from models.features import build_dataset, get_features
from pathlib import Path
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification


def run_model(df: pd.DataFrame):

    output_dir = Path("../models/fine_tuned_model/").resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_exists = (output_dir / 'tokenizer_config.json').is_file()

    train = True
    if model_exists: # If model already saved, load it
        train = False # no need to train
        model = BertForSequenceClassification.from_pretrained(output_dir).to(device)
        tokenizer = BertTokenizer.from_pretrained(output_dir)
        print("Model and tokenizer loaded from: ", output_dir)

    else:
        print("Loading failed. Creating new model and tokenizer.")
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataloader, val_dataloader = build_dataloader(df, tokenizer, device)
    model.to(device)
    torch.cuda.empty_cache()

    if train:
        train_model(train_dataloader, model, tokenizer, output_dir)

    return evaluate_model(val_dataloader, model)


def train_model(train_dataloader: DataLoader, model: BertForSequenceClassification, tokenizer: BertTokenizer, output_dir: Path):
    """ Fine-tune the model 
    """
    optimizer = AdamW(model.parameters(), lr=1e-5)
    epochs = 4

    for epoch in range(epochs):
        model.train()
        train_dataloader = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
        for batch in train_dataloader:
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()    
    
    model.save_pretrained(output_dir, from_pt=True)
    tokenizer.save_pretrained(output_dir, from_pt=True)

def evaluate_model(val_dataloader: DataLoader, model: BertForSequenceClassification)-> Tuple[List, List]:
    """ evaluate the model 
    """
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc='Validation', leave=False):
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=1)

            predictions.extend(predicted_labels.cpu().numpy())
            true_labels.extend(batch[2].detach().cpu().numpy())

    return predictions, true_labels

def build_dataloader(df: pd.DataFrame, tokenizer: BertTokenizer, device: torch.device)-> Tuple[DataLoader, DataLoader]:
    """ Builds training and validation dataloaders
    """
    dataset = build_dataset(tokenizer, get_features(df), device)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    return train_dataloader, val_dataloader