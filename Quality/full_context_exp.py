from quality_utils import get_data_dev
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
from dtc import *
from dtc.utils import *
import re
import json

qa_model = GPT()

df = get_data_dev()

num_questions = 30

df = df.head(num_questions) 


def experiment():

	num_correct = 0
	total_questions = 0

	results = {}

	for index, row in df.iterrows():

		passage = row['passage']
		title = row['title']
		question = row['question']
		options = row['options']
		answer = row['answer']
		formatted_options = ""
		for i, option in enumerate(options, 1):
			formatted_options += f'({chr(64 + i)}) {option} '

		query = f"{question} \\n {formatted_options}"

		context = passage

		match, model_response = get_mcq_answer(question, context, formatted_options, len(options), qa_model)

		if match == 'I' or match == 'N' or '-1' in model_response:
		    match = None
		elif match:
		    match = ord(match) - 64

		if match == row['answer']:

			num_correct += 1

		total_questions += 1

		accuracy = num_correct/total_questions

		print('Match: ', match)
		print('Real: ', row["answer"])

		print('Accuracy: ', accuracy)

		print('-'*50)

		print('\n')

		results[index] = {
			"question": question,
			"options": options,
			"model_response": model_response,
			"match": match,
			"accuracy": accuracy
		}

		with open("gpt_3_turbo_full_context.json", "w") as f:
			json.dump(results, f)

	
experiment()


