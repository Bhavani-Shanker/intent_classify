import streamlit as st
import pandas as pd
from transformers import pipeline

# Initialize Streamlit app
st.title("Zero-Shot Classification with BART")

# Define the sequence and candidate labels
sequence_to_classify = st.text_input("Enter the text to classify:", "Get the legislation details on minorities")
candidate_labels = st.text_input("Enter the candidate labels (comma-separated):", "conventions, legislation, Generic")
candidate_labels = [label.strip() for label in candidate_labels.split(",")]

if st.button("Classify"):
    # Initialize the classifier
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    # Get the classification results
    result = classifier(sequence_to_classify, candidate_labels)

    # Find the label with the maximum score
    max_score_index = result['scores'].index(max(result['scores']))
    max_score_label = result['labels'][max_score_index]
    max_score = result['scores'][max_score_index]

    # Create a DataFrame
    data = {
        'Comment': [sequence_to_classify],
        'Max Score': [max_score],
        'Corresponding Class': [max_score_label]
    }

    df = pd.DataFrame(data)

    # Display the DataFrame
    st.write("Classification Result:")
    st.dataframe(df)
