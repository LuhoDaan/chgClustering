# CHG Emergency Clustering

This Python script clusters similar emergency change requests (CHG emergencies) extracted from ServiceNow using TF-IDF vectorization and cosine similarity to help identify common themes or issues that recur across different incidents.
This will allow to assist in the analytical review of CHG emergencies, accellerating the identification of recurring incidents. By clustering similar entries, the script facilitates more efficient resource allocation and targeted problem-solving efforts within MICS and PA.

## Features

- **Text Preprocessing**: Converts text descriptions into a simplified, tokenized Bag of Words format.
- **TF-IDF Vectorization**: Transforms text data into numerical format to measure textual similarity.
- **Cosine Similarity Scoring**: Calculates similarity scores between all pairs of text entries.
- **Clustering**: Groups similar CHG emergencies based on similarity scores.
- **Duplicate Handling**: Removes duplicate or subset groups within the clusters.

## Extraction
The script processes input from an Excel file generated through a ServiceNow extraction. Performing this extraction requires PA account privileges, specifically at the power user level.

The data extraction from ServiceNow is configured via a query that specifies the timeframe for the data. For this project, the query was set to include all CHG emergencies that were opened and closed from January 2023 onward. This timeframe was chosen to ensure the analysis is current.


### Query Example:

All>Change Type = Emergency>Configuration Item Custodian = ### (L048819) .or. Configuration Item Custodian = ### .or. Configuration Item Custodian = ### .or. Configuration Item Custodian = ### >Closed > 2023-01-01 00:00:00>Opened > 2023-01-01 00:00:00

**Simply substitute above the ### with the secondary PA names**


## Installation

### Prerequisites

Before you can run this script, you need to install Python and the necessary Python libraries. The easiest way to manage the libraries is by using `pip`. This project requires the following libraries:

- pandas
- numpy
- nltk
- scikit-learn
- unidecode

## Usage

To run the script, navigate to the script's directory and execute:
```
python chgemergency.py
```
After running you'll obtain an Excel file 'excel_label_removed_duplicates.xlsx' containing the clustered sentences.

Make sure to update the INPUT_EXTRACTION variable in the script to point to the path of your ServiceNow extraction Excel file.

### Configuration
- **INPUT_EXTRACTION**: Path to the Excel file containing CHG emergencies data.
- **similarity_treshold**: Threshold for considering sentences as similar (default is 0.7).
