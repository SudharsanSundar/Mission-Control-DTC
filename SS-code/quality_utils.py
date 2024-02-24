# TODO: Salman: get this working with the eval script

import json
import pandas as pd
import re


def process_options(options):

    output = []

    for i in range(len(options)):

        output.append(f"{i+1}. " + options[i])

    return ' '.join(output)


def append_to_file(question_id, answer, file_name):
    with open(file_name, 'a') as file:
        file.write(f"{question_id},{answer}\n")


def get_data_test():
    # Create an empty DataFrame with the required columns
    df = pd.DataFrame(columns=['title', 'passage', 'question', 'options', 'question_id', 'difficult'])

    # Read the data from the file
    with open('data/v1.0.1/QuALITY.v1.0.1.htmlstripped.test', 'r') as f:
        for line in f:
            data = json.loads(line)
            title = data['title']
            remove_spaces = lambda text: re.sub(r'\s+', ' ', text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' '))
            passage = remove_spaces(data['article'])

            for question_dict in data['questions']:
                question = question_dict['question']
                options = question_dict['options']
                question_id = question_dict['question_unique_id']
                difficult = question_dict['difficult']

                # Append the data to the DataFrame
                new_row = pd.DataFrame({
                    'title': [title],
                    'passage': [passage],
                    'question': [question],
                    'options': [process_options(options)],
                    'question_id': [question_id],
                    'difficult': [difficult]
                })
                df = pd.concat([df, new_row], ignore_index=True)


    return df


def get_data_dev():
    # Create an empty DataFrame with the required columns
    df = pd.DataFrame(columns=['title', 'passage', 'question', 'options', 'question_id', 'difficult'])

    # Read the data from the file
    with open('data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev', 'r') as f:
        for line in f:
            data = json.loads(line)
            title = data['title']
            remove_spaces = lambda text: re.sub(r'\s+', ' ', text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' '))
            passage = remove_spaces(data['article'])

            for question_dict in data['questions']:
                question = question_dict['question']
                options = question_dict['options']
                answer = question_dict['gold_label']
                difficult = question_dict['difficult']

                # Append the data to the DataFrame
                new_row = pd.DataFrame({
                    'title': [title],
                    'passage': [passage],
                    'question': [question],
                    'options': [options],
                    'answer': [answer],
                    'difficult': [difficult]
                })
                df = pd.concat([df, new_row], ignore_index=True)


    return df



            



        

        

        
    


















