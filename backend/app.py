import pandas as pd
import streamlit as st

from src.data.transform import transform_data
from src.models.bert_classifier import run_model
from sklearn.metrics import accuracy_score


from pathlib import Path

def get_files_in_directory(directory):
    directory_path = Path(directory)
    files = [file.name for file in directory_path.iterdir() if file.is_file()]
    return files


# Function to load and run the model on a CSV file
def run_model_on_csv(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Run your model on the DataFrame (replace this with your actual model code)
    ts_df = transform_data(df)
    predictions, true_labels = run_model(ts_df)
    # Return results or display them in Streamlit
    return predictions, true_labels

def main():
    st.title("Model Runner App")

    # Get files in /data/raw/ directory
    data_dir = Path("data/raw").resolve()
    if data_dir.exists() and data_dir.is_dir():
        files = [file.name for file in data_dir.iterdir() if file.is_file()]

        # Display files in a selectbox
        selected_file = st.selectbox("Select a CSV file:", files)

        if selected_file:
            file_path = data_dir / selected_file

            # Run the model on the selected CSV file
            predictions, true_labels = run_model_on_csv(file_path)
            accuracy = accuracy_score(true_labels, predictions)

            # Display results or any other information
            st.write("Model Results:")
            st.write("Predictions:", predictions)
            st.write("True Labels:", true_labels)
            st.write("Accuracy:", accuracy)


if __name__ == "__main__":
    main()