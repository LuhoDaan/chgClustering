# -*- coding: utf-8 -*-
"""
This script applies information retrieval techniques to cluster similar emergency change requests (CHG emergencies) extracted from ServiceNow. 
It employs TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to transform the textual data into a numerical format, enhancing the 
ability to measure textual similarity through cosine similarity scores.

The workflow includes:
1. Preprocessing the data to convert text descriptions into a simplified, tokenized 'Bag of Words' format.
2. Vectorizing the processed text using TF-IDF to capture the importance of terms within documents and across a corpus of documents.
3. Calculating normalized cosine similarity scores to determine the similarity between different CHG emergencies.
4. Clustering similar CHG emergencies based on a defined similarity threshold to group related incidents.
5. Handling potential duplicates or subsets within clusters to ensure each group maintains unique elements.
6. Exporting the results to an Excel file for further analysis and to create a dashboard in PowerBi useful for Process Automation Staff.

This script is designed to assist in the analytical review of CHG emergencies, allowing for the identification of common themes or issues that 
recur across different incidents. By clustering similar entries, the script facilitates more efficient resource allocation and targeted problem-solving 
efforts within MICS and PA.

Dependencies:
- pandas for data manipulation and reading/writing Excel files.
- nltk for natural language processing, including stopwords removal.
- numpy for numerical operations.
- scikit-learn for machine learning operations including TF-IDF vectorization and cosine similarity calculations.
- re and unidecode for regular expressions and text cleaning.

Usage:
Ensure that the INPUT_EXTRACTION variable is set to the path of the Excel file containing the CHG emergency data from ServiceNow before running the script.
"""

import pandas as pd
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import re
from unidecode import unidecode


INPUT_EXTRACTION = "PATH TO YOUR EXCEL FILE COMING FROM A SERVICE NOW EXTRACTION"

# Preprocessing
# -------------
# This section handles the cleaning and preparation of text data, converting it to a bag of words representation.

nltk.download('stopwords')


def data_cleaning_bow(dataframe):
    """Converts specified text columns in a dataframe to a Bag of Words representation.

    This function adds a new column 'BoW' to the dataframe, representing a simplified, tokenized form of text from the 'Short Description' and 'Description' columns.
    It processes the text by removing non-informative words and stopwords, then tokenizes the cleaned text into a bag of words format.

    Parameters
    ----------
    dataframe : DataFrame
        Pandas DataFrame containing the columns 'Short Description' and 'Description'.

    Returns
    -------
    DataFrame
        Modified DataFrame with an additional column 'BoW'.
    """
    def text_cleaning(text_raw):
        """Removes non-informative and common words from text, returning significant tokens.

        This function filters out common words and stopwords from the input text, lowers the case, removes special characters and accents, and finally tokenizes the cleaned text. The result is a list of words deemed significant for further analysis.

        Parameters
        ----------
        text_raw : str
            Raw text from which to remove stopwords and non-informative words.

        Returns
        -------
        list
            List of significant words (tokens) from the cleaned text.
        """

        sequence_to_remove = ['descrizione del problema', 'analisi e soluzione','modifica','emergenza','intervento','eseguito','produzione','appeso','-', 'EMERGENCY', 
                              'DATA', 'SUPPORT', 'MICS-SHARED', 'macchina', 'PA', '"Linea', 'Modifica', 'MI&CS', 'di', 'Filling', 'Problema', 'su', 'MICS', 'PFS', 'per', 
                              'SESTO', 'Esecuzione', 'Emergency','<']
        stop_words = set(nltk.corpus.stopwords.words('italian') + [])
        text_filtered = text_raw.lower()  
        # remove text after '***Dati Modificati***'
        text_filtered = re.sub('\n.*'+'dati modificati'+'.*', ' ', text_filtered)
        # remove section separators
        for el in sequence_to_remove:
            text_filtered = re.sub(r'\b{}\b'.format(re.escape(el)), '', text_filtered)
        # clean and standardize the text
        text_filtered = unidecode(text_filtered)  # remove accents
        text_filtered = re.sub('[^A-Za-z0-9_ ]+', ' ', text_filtered)
        words = nltk.word_tokenize(text_filtered, language="italian")  # tokenize
        filtered_sentence = [w for w in words if w not in stop_words]  # remove stopwords
        return filtered_sentence

    def extract_bag_of_words(row):
        input_text = row["Short Description"] + " " + row["Description"]
        input_cleaned = text_cleaning(input_text)
        return ' '.join(input_cleaned)

    dataframe["BoW"] = dataframe.apply(extract_bag_of_words, axis=1)
    return dataframe

# Vectorization
# -------------
# This section focuses on converting the processed text into a numerical format using the TF-IDF vectorization technique.

def get_test_scores(vectorizer,matrix_bucket):

    """Calculates normalized cosine similarity scores for a list of sentences.

    This function vectorizes a list of sentences using the provided TfidfVectorizer, and then computes the cosine similarity between every pair of vectorized sentences.
    It returns a normalized score matrix where the diagonal elements represent self-similarity (always 1) and off-diagonal elements indicate the similarity between different
    sentences, scaled to range from 0 to 1. A score of 1 indicates identical sentences, and a score of 0 indicates no similarity.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        A fitted TF-IDF vectorizer for transforming text to vector form.
    matrix_bucket : list of str
        A list of sentences to be vectorized and compared.

    Returns
    -------
    scores_normed : numpy.ndarray
        A 2D array of cosine similarity scores between all pairs of sentences, normalized to the range [0, 1].
    """
    tfidf_matrix_bucket = vectorizer.transform(matrix_bucket)
    scores = cosine_similarity(tfidf_matrix_bucket, tfidf_matrix_bucket)
    scores_normed = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    return scores_normed

def eliminate_duplicates(df):
    """Removes groups with duplicated entries based on CHG EMERGENCY 'Number' values within each 'labels' group.

    This function identifies and removes groups that contain identical 'Number' entries or are subsets of larger groups, leaving just one instance of each.
    It first sorts groups by the count of unique 'Number' entries in descending order to efficiently determine subset relationships.
    Each group is then compared to others, and subsets are marked for removal.

    Parameters
    ----------
    df : DataFrame
        DataFrame labeled with repeated groups, each group identified by a unique 'labels' value.

    Returns
    -------
    DataFrame
        A DataFrame with duplicate or subset groups removed, retaining only unique sets of 'Number' entries for each label.
    """

    grouped_df = df.groupby('labels', as_index=False)
    groups = [group for _, group in grouped_df]
    groups.sort(key=lambda x: len(set(x['Number'])), reverse=True)

    to_remove = []
    for i in range(len(groups) - 1):
        if i in to_remove:
            continue
        for j in range(i + 1, len(groups)):
            if j in to_remove:
                continue
            set_i = set(groups[i]['Number'])
            set_j = set(groups[j]['Number'])

            if set_j.issubset(set_i):
                to_remove.append(j)

            elif set_i.issubset(set_j):
                to_remove.append(i)
                break  # No need to compare i with other groups if it's already marked for removal

    final_groups = [group for index, group in enumerate(groups) if index not in to_remove]

    result_df = pd.concat(final_groups).reset_index(drop=True)
    return result_df

def clusterize(df_with_labels, df_clean, scores, similarity_threshold):
    """Clusters sentences based on cosine similarity scores above a given threshold.

    This function assigns a unique label to clusters of sentences from `df_clean` based on their similarity scores. 
    Sentences with scores exceeding the `similarity_threshold` are grouped together. The function manages potential duplicates post-clustering using `eliminate_duplicates`.

    Parameters
    ----------
    df_with_labels : DataFrame
        DataFrame to store labeled sentences, allowing repeated entries.
    df_clean : DataFrame
        DataFrame containing the 'BoW' text representation.
    scores : numpy.ndarray
        2D array of cosine similarity scores, normalized to the range [0, 1].
    similarity_threshold : float
        Threshold for considering sentences as similar (range 0 to 1).

    Returns
    -------
    DataFrame
        Updated `df_with_labels` with new clusters labeled.

    """

    df_clean['labels'] = 'Initial'
    cluster_label_counter = 0  

    for i in range(len(df_clean)):
        similar_indices = np.where(scores[:, i] > similarity_threshold)[0]
        if len(similar_indices) > 0:
            cluster_label = f'cluster_{cluster_label_counter}'
            df_clean.loc[similar_indices, 'labels'] = cluster_label
            cluster_label_counter += 1  
        df_with_labels = pd.concat([df_with_labels, df_clean.loc[similar_indices].drop('BoW', axis=1)], ignore_index=True)

    print(f"BEFORE DUPLICATES REMOVAL: {df_with_labels.shape[0]}")
    df_with_labels = eliminate_duplicates(df_with_labels)
    print(f"AFTER DUPLICATES REMOVAL: {df_with_labels.shape[0]}")

    return df_with_labels



# Main function
# -------------
# This section focuses on orchestrating the utility functions to extract the relevant clusters 

def main():
    """Main execution function for clustering similar sentences within each configuration items.

    This function performs several steps to process and cluster sentences from an Excel sheet:
    1. Loads the data from an Excel file.
    2. Orders the 'Configuration Item' based on their occurrence frequency.
    3. For each unique configuration item, it filters the data, cleans it using a Bag of Words model, and then applies TF-IDF vectorization.
    4. Computes cosine similarity scores for the sentences.
    5. Clusters sentences based on a predefined similarity threshold, handling edge cases for items with only one sentence.
    6. Concatenates all the processed data into a single DataFrame and exports it to an Excel file.

    Outputs:
    - An Excel file 'excel_label_removed_duplicates.xlsx' containing the clustered sentences.
    """    

    df = pd.read_excel(INPUT_EXTRACTION,sheet_name='Page 1')
    list_of_machines_ordered = list(df['Configuration Item'].value_counts().index)
    similarity_threshold = 0.7
    df_list_per_machine = []
    df_list_labels_per_machine = []

    for i, machine_name in enumerate(list_of_machines_ordered):
        df_filtered = df[df['Configuration Item'] == machine_name].copy(deep=True).reset_index(drop = True)
        df_list_per_machine.append(df_filtered)

    i= 0
    for df_raw in df_list_per_machine:
        i+=1
        df_clean = data_cleaning_bow(df_raw)
        df_with_labels = pd.DataFrame(columns = df_clean.columns)
        #Initialize Vectorizer fitted for each ci's vocabulary only, this can be adjusted to fit the whole corpus instead.
        vectorizer = TfidfVectorizer().fit(list(df_clean['BoW']))
        scores = get_test_scores(vectorizer, list(df_clean['BoW']))
        print(f"Machine {i}")
        print(f"BEFORE CLUSTERING{df_clean.shape[0]}")
        if df_clean.shape[0] == 1:
            df_with_labels = df_clean.copy(deep=True).reset_index(drop = True)
            df_with_labels['labels'] = 'Gruppo_0'
        else:
            if i == 1:
                df_with_labels = clusterize(df_with_labels,df_clean,scores,0.825)
            else:
                df_with_labels = clusterize(df_with_labels,df_clean,scores,similarity_threshold)
        df_list_labels_per_machine.append(df_with_labels)



    df_only = pd.concat(df_list_labels_per_machine,ignore_index=True)
    df_only.to_excel('excel_label_removed_duplicates.xlsx')


main()

