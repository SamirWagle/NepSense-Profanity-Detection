import os
import re
import requests
import pickle
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from tqdm import tqdm
from prettytable import PrettyTable
from langdetect import detect, LangDetectException
from ai4bharat.transliteration import XlitEngine
from transformers import BertTokenizer, BertModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
from nltk.util import ngrams


class Download:
    def __init__(self):
        pass

    def download_file_from_google_drive(self, file_id, destination):
        """Download file from Google Drive."""
        def get_confirm_token(response):
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value
            return None

        def save_response_content(response, destination):
            CHUNK_SIZE = 32768
            total_size = int(response.headers.get('content-length', 0))
            with open(destination, "wb") as f, tqdm(
                desc=destination,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params={'id': file_id}, stream=True)
        token = get_confirm_token(response)

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        save_response_content(response, destination)


# Instantiate the Download class
downloader = Download()

# Dictionary containing filenames and their corresponding Google Drive file IDs
file_ids = {
    'Multi_Model_Multi_Output.h5': '1L8V2by-UzjxPSL7NyZckTkarSt8AHGBw',
    'Multi_Model_Multi_Output.pkl': '1P5_Nz52s_Wokuymp26aPOp9GgOe8nuXF',
    'Mutlilabel_LSTM_Offensive_Profane.h5': '1bwOOub0XeazLwXisQ981EDMXuvsQvHhE',
    'Mutlilabel_LSTM_Offensive_Profane.pkl': '18xonXzSUxkpE1NS4dJnZR37uySvNG2B6',
    'Binomial_LSTM_Profane.h5': '1dr87w_4iWdiV2EUOGi4asHF1RNtWeS4f',
    'Binomial_LSTM_Profane.pkl': '1kJTe25qdBu6q5i6Gk5VYtDAA06Puj9Ml',
    'Binomial_LSTM_Offensive.keras': '1rDBl9_WvaA7YWNx_sZ08BQFGTTKCk5wS',
    'Binomial_LSTM_Offensive.pkl': '1UdLUoalPqotM5tYHwYwZDXffp0mbnLZc',
    "Emoji Sheets - Emoji Only.csv": "1bHK-ofASD0XC-Z1gtbKbalWj6B_71umw"
}

# Specify your download directory
download_dir = './downloaded_files/'
os.makedirs(download_dir, exist_ok=True)


# Load models and tokenizers


def load_model_and_tokenizer(model_path, tokenizer_path):
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


# Initialize BERT model and tokenizer
model_name = "bert-base-multilingual-cased"
tokenizer_bert = BertTokenizer.from_pretrained(model_name)
model_bert = BertModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_bert.to(device)
model_bert.eval()

# Function to remove emojis from text


def remove_emojis(text):
    emoji_df = pd.read_csv(os.path.join(
        download_dir, 'Emoji Sheets - Emoji Only.csv'))
    emoji_list = emoji_df['Emoji_List'].tolist()
    pattern = '[' + ''.join(f'\\U{cp[1:]:0>8}' for cp in emoji_list) + ']'
    return re.compile(pattern, re.UNICODE).sub(r'', text)

# Function to preprocess text for model input


def preprocess_text(text, tokenizer):
    sequences = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequences, padding='post', maxlen=500)

# Function to predict labels using a multi-output model


def predict_labels(text):
    tokenized_text = tokenizer_bert.encode(text, add_special_tokens=True)
    text_ngrams = list(ngrams(tokenized_text, 2))
    embeddings = []

    for gram in text_ngrams:
        input_ids = torch.tensor(gram).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model_bert(input_ids)
        embeddings.append(torch.mean(
            outputs.last_hidden_state, dim=1).squeeze().cpu().numpy())

    embeddings_array = np.array(embeddings)
    multi_model, multi_tokenizer = load_model_and_tokenizer(
        os.path.join(download_dir, 'Multi_Model_Multi_Output.h5'),
        os.path.join(download_dir, 'Multi_Model_Multi_Output.pkl')
    )
    max_len = multi_model.input_shape[1]
    padded_embeddings = pad_sequences(
        [embeddings_array], maxlen=max_len, padding='post', dtype='float32')
    predictions = multi_model.predict(padded_embeddings)
    gender_label = int(predictions[0] > 0.5)
    profanity_label = np.argmax(predictions[1], axis=1)[0]
    pred_accuracy = predictions[1][0][profanity_label]
    return gender_label, profanity_label, pred_accuracy

# Function to predict labels using a multilabel model


def predict_multilabel(text):
    multilabel_lstm_model, multilabel_tokenizer = load_model_and_tokenizer(
        os.path.join(download_dir, 'Mutlilabel_LSTM_Offensive_Profane.h5'),
        os.path.join(download_dir, 'Mutlilabel_LSTM_Offensive_Profane.pkl')
    )
    preprocessed_text = preprocess_text(text, multilabel_tokenizer)
    prediction = multilabel_lstm_model.predict(preprocessed_text)
    profanity_pred = np.argmax(prediction, axis=1)[0]
    pred_accuracy = prediction[0][profanity_pred]
    return profanity_pred, pred_accuracy

# Function to predict binary labels using a binomial model


def predict_binomial(text, model, tokenizer):
    input_sequence = tokenizer.texts_to_sequences([text])
    input_padded = pad_sequences(input_sequence, maxlen=500, padding='post')
    prediction = model.predict(input_padded)
    predicted_label = int(prediction.argmax(axis=-1)[0])
    pred_accuracy = prediction[0][predicted_label]
    return predicted_label, pred_accuracy

# e = XlitEngine(["ne"], beam_width=10, src_script_type="en")


@st.cache(allow_output_mutation=True)
def create_XlitEngine():
    return XlitEngine(["ne"], beam_width=10, src_script_type="en")


# Create the XlitEngine instance
e = create_XlitEngine()


def nepali_nlp_text_conversion(text):
    # Function to convert text to Nepali using transliteration
    e = XlitEngine(["ne"], beam_width=10, src_script_type="en")
    if text.strip():
        try:
            if detect(text) != "ne":
                return e.translit_sentence(text)["ne"]
        except LangDetectException:
            pass
    return text

# Function to print results in a PrettyTable format


def print_results(user_text, gender_label, multi_profanity_label, multi_pred_accuracy,
                  multilabel_profanity_label, multilabel_pred_accuracy,
                  binomial_profanity, binomial_profanity_pred_accuracy,
                  binomial_offensive, binomial_offensive_pred_accuracy):
    gender = 'Male' if gender_label == 1 else 'Female'
    multi_profanity_classes = {
        0: 'Non-Offensive', 1: 'Offensive', 2: 'Profane'}

    # Create the additional information table with the required format
    additional_info_table = PrettyTable()
    additional_info_table.field_names = ["--------Test Result Of--------"]
    # Add the user's text to the table
    additional_info_table.add_row(
        [f"                              {user_text}                            "])

    # Display the results in a table
    table = PrettyTable()
    table.field_names = ["Model", "Gender",
                         "Profanity/Offensiveness", "Pred Accuracy"]
    table.add_row(["Multi-Output Model", gender,
                  multi_profanity_classes[multi_profanity_label], f'{multi_pred_accuracy:.4f}'])
    table.add_row(["Multilabel LSTM Model", "-",
                  multi_profanity_classes[multilabel_profanity_label], f'{multilabel_pred_accuracy:.4f}'])
    table.add_row(["Binomial LSTM Profanity Model", "-",
                  {0: 'Non-Profane', 1: 'Profane'}[binomial_profanity], f'{binomial_profanity_pred_accuracy:.4f}'])
    table.add_row(["Binomial LSTM Offensive Model", "-",
                   {0: 'Non-Offensive', 1: 'Offensive'}[
                       binomial_offensive], f'{binomial_offensive_pred_accuracy:.4f}'])

    # Print the tables
    print(additional_info_table)
    print(table)

    data = {
        'Model': [
            'Multi-Output Model',
            'Multilabel LSTM Model',
            'Binomial LSTM Profanity Model',
            'Binomial LSTM Offensive Model'
        ],
        'Gender': [
            'Male' if gender_label == 1 else 'Female',
            '-',
            '-',
            '-'
        ],
        'Profanity/Offensiveness': [
            multi_profanity_classes[multi_profanity_label],
            multi_profanity_classes[multilabel_profanity_label],
            {0: 'Non-Profane', 1: 'Profane'}[binomial_profanity],
            {0: 'Non-Offensive', 1: 'Offensive'}[
                binomial_offensive]
        ],
        'Pred Accuracy': [
            round(multi_pred_accuracy, 4),
            round(multilabel_pred_accuracy, 4),
            round(binomial_profanity_pred_accuracy, 4),
            round(binomial_offensive_pred_accuracy, 4),
        ]
    }

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)

    # Display the DataFrame as a table in Streamlit
    st.subheader(f'Predictions of {user_text}')
    col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 1])
    col1.write('**SN**')
    col2.write('**Model**')
    col3.write('**Gender**')
    col4.write('**Profanity/Offensiveness**')
    col5.write('**Pred Accuracy**')

    for i in range(len(df)):
        col1.write(i + 1)
        col2.write(df['Model'][i])
        col3.write(df['Gender'][i])
        col4.write(df['Profanity/Offensiveness'][i])
        col5.write(df['Pred Accuracy'][i])


def add_data_to_excel(user_text, gender, multi_profanity, multi_pred_accuracy,
                      multilabel_profanity, multilabel_pred_accuracy, binomial_profanity,
                      binomial_profanity_pred_accuracy, binomial_offensive,
                      binomial_offensive_pred_accuracy, user_responses_profanity, user_responses_gender,
                      tested_file_name='testeddata.xlsx'):

    # Try to import the Excel file
    try:
        if os.path.exists(tested_file_name):
            # Read the existing Excel file into a DataFrame
            tested_df = pd.read_excel(tested_file_name, index_col=False)
            print("Excel file imported successfully.")
        else:
            # If the file doesn't exist, create a new DataFrame
            tested_df = pd.DataFrame(columns=[
                'user_text', 'gender', 'multi_profanity', 'multi_pred_accuracy',
                'multilabel_profanity', 'multilabel_pred_accuracy',
                'binomial_profanity', 'binomial_profanity_pred_accuracy',
                'binomial_offensive', 'binomial_offensive_pred_accuracy',
                "user_responses_profanity", "user_responses_gender",
            ])
            print("Excel file not found. New DataFrame created.")

    except FileNotFoundError:
        print(
            f"FileNotFoundError: The file '{tested_file_name}' was not found.")
        return

    # Create a dictionary with the data
    new_row = {
        'user_text': [user_text],
        'gender': [gender],
        'multi_profanity': [multi_profanity],
        'multi_pred_accuracy': [multi_pred_accuracy],
        'multilabel_profanity': [multilabel_profanity],
        'multilabel_pred_accuracy': [multilabel_pred_accuracy],
        'binomial_profanity': [binomial_profanity],
        'binomial_profanity_pred_accuracy': [binomial_profanity_pred_accuracy],
        'binomial_offensive': [binomial_offensive],
        'binomial_offensive_pred_accuracy': [binomial_offensive_pred_accuracy],
        'user_responses_profanity': [user_responses_profanity],
        'user_responses_gender': [user_responses_gender],
    }

    # Create a new DataFrame from the dictionary
    new_row_df = pd.DataFrame(new_row)

    # Assign the new row to the DataFrame
    tested_df.loc[len(tested_df)] = new_row_df.iloc[0].values
    tested_df.to_excel(tested_file_name, index=False)
    print(f"Data added and saved to '{tested_file_name}' successfully.")


def main():
    st.set_page_config(
        page_title="Nepsense",
        page_icon=":chart_with_upwards_trend:",  # Emoji icon example
        layout="wide",  # Wide layout
        initial_sidebar_state="auto",  # Sidebar starts in auto-collapse mode
    )
    st.title("NepSense - Profanity and Gender Prediction")
    st.write("Enter a sentence to predict its gender and profanity classification:")

    user_text = st.text_area("Enter text:", "")

    # Creating columns for dropdowns
    col1, col2 = st.columns(2)
    with col1:
        user_gess_profanity = st.selectbox("Guess the profanity level of the entered text:                           .", [
            "None", "Offensive", "Profane"])

    with col2:
        user_gess_gender = st.selectbox("Guess the gender of the person who might have written/said the entered text:", [
            "Male", "Female", "Both"])
    if st.button("Predict"):
        if not user_text or not user_text.strip():
            st.error("Please enter some text.")
        else:
            user_text = nepali_nlp_text_conversion(user_text)
            user_text = remove_emojis(user_text)

            # Download each file
            for filename, file_id in file_ids.items():
                destination_path = os.path.join(download_dir, filename)
                if not os.path.exists(destination_path):
                    print(f"Downloading {filename}...")
                    downloader.download_file_from_google_drive(
                        file_id, destination_path)
                else:
                    print(f"File {filename} already downloaded.")

            gender_label, multi_profanity_label, multi_pred_accuracy = predict_labels(
                user_text)
            multilabel_profanity_label, multilabel_pred_accuracy = predict_multilabel(
                user_text)

            binomial_lstm_model_profanity, binomial_tokenizer_profanity = load_model_and_tokenizer(
                os.path.join(download_dir, 'Binomial_LSTM_Profane.h5'),
                os.path.join(download_dir, 'Binomial_LSTM_Profane.pkl')
            )

            binomial_profanity, binomial_profanity_pred_accuracy = predict_binomial(
                user_text, binomial_lstm_model_profanity, binomial_tokenizer_profanity)

            binomial_lstm_model_offensive, binomial_tokenizer_offensive = load_model_and_tokenizer(
                os.path.join(download_dir, 'Binomial_LSTM_Offensive.keras'),
                os.path.join(download_dir, 'Binomial_LSTM_Offensive.pkl')
            )
            binomial_offensive, binomial_offensive_pred_accuracy = predict_binomial(
                user_text, binomial_lstm_model_offensive, binomial_tokenizer_offensive)

            print_results(user_text, gender_label, multi_profanity_label, multi_pred_accuracy,
                          multilabel_profanity_label, multilabel_pred_accuracy,
                          binomial_profanity, binomial_profanity_pred_accuracy,
                          binomial_offensive, binomial_offensive_pred_accuracy)

            add_data_to_excel(
                user_text=user_text,
                gender=gender_label,
                multi_profanity=multi_profanity_label,
                multi_pred_accuracy=multi_pred_accuracy,
                multilabel_profanity=multilabel_profanity_label,
                multilabel_pred_accuracy=multilabel_pred_accuracy,
                binomial_profanity=binomial_profanity,
                binomial_profanity_pred_accuracy=binomial_profanity_pred_accuracy,
                binomial_offensive=binomial_offensive,
                binomial_offensive_pred_accuracy=binomial_offensive_pred_accuracy,
                user_responses_profanity=user_gess_profanity,
                user_responses_gender=user_gess_gender)


if __name__ == "__main__":
    main()
