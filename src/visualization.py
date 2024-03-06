import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path


def plot_confusion_matrix(true_labels, predictions):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    output_path = Path('../reports/')
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / 'confusion_matrix_plot.png'
    plt.savefig(output_file)
    plt.close()


def plot_label_counts(df: pd.DataFrame):
    plt.pie(df['label'].value_counts(), labels=['not sustainable', 'sustainable'], autopct="%0.2f")
    output_path = Path('../reports/eda/')
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / 'label_counts_pie_chart.png'
    plt.savefig(output_file)
    plt.close()


def plot_num_of_stats(df: pd.DataFrame, column_name: str):
    sns.histplot(df[df['label'] == 0][column_name])
    sns.histplot(df[df['label'] == 1][column_name], color='red')
    output_path = Path('../reports/eda/')
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f'{column_name}_chart.png'
    plt.savefig(output_file)
    plt.close()
