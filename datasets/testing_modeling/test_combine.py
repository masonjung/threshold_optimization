# combine test datasets with dataset names

import pandas as pd

# read datasets

dataset_detectors = [PATH]
test_dataset_three_RoBERTa_detectors = pd.read_csv(dataset_detectors)

test_dataset_three_RoBERTa_detectors.columns


dataset_kaggle = [PATH]
dataset_mage = [PATH]
dataset_roft = [PATH]
dataset_semeval = [PATH]

dataset1 = pd.read_csv(dataset_kaggle)
dataset2 = pd.read_csv(dataset_mage)
dataset3 = pd.read_csv(dataset_roft)
dataset4 = pd.read_csv(dataset_semeval)

# print all columns in datset1 to 4
print("Columns in dataset1:", dataset1.columns)
print("Columns in dataset2:", dataset2.columns)
print("Columns in dataset3:", dataset3.columns)
print("Columns in dataset4:", dataset4.columns)


def add_datasource_label(df, datasets, labels, text_columns):
    # Initialize an empty 'source' column in the dataframe
    df['source'] = ''

    for dataset, label, text_column in zip(datasets, labels, text_columns):
        dataset_df = pd.read_csv(dataset)

        # Ensure the column exists in the dataset
        if text_column not in dataset_df.columns:
            raise ValueError(f"Column '{text_column}' not found in the dataset '{dataset}'. Please provide the correct column name.")
        
        # Find matching essays and add the label to the 'source' column
        df['source'] = df.apply(lambda row: row['source'] + ',' + label if row['essay'] in dataset_df[text_column].values else row['source'], axis=1)
    
    # Clean up the 'source' column, removing leading commas
    df['source'] = df['source'].str.lstrip(',')
    return df

datasets = [dataset_kaggle, dataset_mage, dataset_roft, dataset_semeval]
labels = ['kaggle', 'mage', 'roft', 'semeval']
text_columns = ['text', 'text', 'prompt_body', 'text']

# Add the 'source' column to the dataframe
test_dataset_source_three_RoBERTa_detectors = add_datasource_label(test_dataset_three_RoBERTa_detectors, datasets, labels, text_columns)

# Print the updated dataframe to check the 'source' column
print(test_dataset_source_three_RoBERTa_detectors[['essay', 'source']])


test_dataset_source_three_RoBERTa_detectors["source"].value_counts()

# Save the updated dataframe to a new CSV file
test_dataset_source_three_RoBERTa_detectors.to_csv([PATH])

test_dataset_three_RoBERTa_detectors.columns