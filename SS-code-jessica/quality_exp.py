from quality_utils import get_data_dev
import pandas as pd

df = get_data_dev()

for index, row in df.iterrows():
    print("Title:", row['title'])
    print("Passage:", row['passage'])
    print("Question:", row['question'])
    print("Options:", row['options'])
    print("Answer:", row['answer'])
    print("Difficult:", row['difficult'])
    print("----------")

    