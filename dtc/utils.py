import re
import tiktoken
import numpy as np

"""
Default function import from Salman/Raptor.

Takes str of text. 
Uses regex to split into whole sentences. 
Packs chunks up to ~max_size with whole sentences.
Returns chunks
"""
def split_text(text, max_tokens, tokenizer=tiktoken.get_encoding("cl100k_base")):
    # Split the text into sentences using regular expressions
    sentences = re.split('(?<=[.!?]) +', text)
    # Count the number of tokens in each sentence
    n_tokens = [len(tokenizer.encode(sentence)) for sentence in sentences]
    print(f'There are this many tokens: {len(tokenizer.encode(text))}')

    chunks = []  # List to hold chunks of text
    tokens_so_far = 0  # Counter for tokens in the current chunk
    chunk = []  # List to hold sentences for the current chunk

    # Loop through each sentence and its token count
    for sentence, token in zip(sentences, n_tokens):

        # Check if adding the current sentence would exceed the max tokens for the chunk
        if tokens_so_far + token > max_tokens:
            chunks.append(" ".join(chunk).strip())
            chunk = []
            tokens_so_far = 0

        # Skip sentences that themselves exceed the max token count
        if token > max_tokens:
            continue

        # Add the current sentence to the chunk
        chunk.append(sentence)
        tokens_so_far += token

    # Add any remaining sentences as a final chunk
    if chunk:
        chunks.append(" ".join(chunk).strip())

    return chunks


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



def extract_letter(answer):
    match = re.search(r'\b([A-Z])\b', answer)
    return match.group(1) if match else None


def get_mcq_answer(question: str, context: str, options: str, num_options: int, qa_model):
    prompt_1 = f'''
    Context: {context}
    Question: {question}
    Options: {options}
    '''

    answer_1 = qa_model.answer_question(prompt_1)

    print('Answer 1: ', answer_1)

    prompt_2 = f'''
    Question: {question}
    Answer: {answer_1}
    Options: {options}
    Thus, amongst options A through {chr(num_options + 64)}, give the letter and nothing else. If none of the options work, write -1
    '''

    answer_2 = qa_model.answer_question(prompt_2)

    print('Answer 2: ', answer_2)

    match = extract_letter(answer_2)

    return match, answer_2



