import pandas as pd
import json

#path
jsonl_path = r'C:\\Users\\minse\Desktop\\Programming\\FairThresholdOptimization\\datasets\\Test_dataset\\subtaskA_train_monolingual.jsonl'

# dataframe
with open(jsonl_path, 'r') as file:
    data = [json.loads(line) for line in file]

df_jsonl = pd.DataFrame(data)


# 1 is AI written and 0 is human written
df_jsonl["label"].value_counts()

# sampling
df_jsonl = df_jsonl.sample(n=3000, random_state=1)

# download location
download_path = r'C:\\Users\\minse\Desktop\\Programming\\FairThresholdOptimization\\datasets\\Test_dataset\\subtaskA_train_monolingual_sampled.csv'

# download dataset
df_jsonl.to_csv(download_path, index=False)

