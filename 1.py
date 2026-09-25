import pandas as pd
import csv
def load_data_and_print(filepath: str,title:str) -> pd.DataFrame:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            train = pd.DataFrame(csv.DictReader(f))
            print(train[title].str.lower().str.replace(r'[^\w\s]', '',regex=True).str.split().explode().value_counts().head(10))
            return train
    except FileNotFoundError:
        print('File Not Found')
load_data_and_print('train_u6lujuX_CVtuZ9i.csv','Property_Area')
