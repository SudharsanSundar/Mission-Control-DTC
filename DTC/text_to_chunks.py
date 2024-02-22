import re
import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")

def split_text(text, tokenizer, max_tokens):
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
 