import pandas as pd
from data.transform import transform_data
from models.bert_classifier import run_model
from visualization import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from data.eda import run_eda_on_data


path = '../data/raw/internship_challenge - dataset.csv'

df = pd.read_csv(path)
fs_df = transform_data(df) # data is read to be used in model

run_eda_on_data(fs_df) # perform EDA on final data

predictions, true_labels = run_model(fs_df) # model is loaded, fine-tuned, evaluated and saved

accuracy = accuracy_score(true_labels, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

plot_confusion_matrix(predictions, true_labels) # Visualization of results
