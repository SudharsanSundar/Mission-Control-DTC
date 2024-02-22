from slidingwindow import GPT
from slidingwindow import SlidingWindow
from text_to_chunks import split_text
from text_to_chunks import tokenizer
# Instantiate the GPT model 
gpt_model = GPT(model="gpt-3.5-turbo") 


file_path = 'livingston.txt'

print("Hello")

# Loading in the Corpus
with open(file_path, 'r') as file:
    data = file.read()
#print(data)
corpus = data 

# Here is the query
query = "How much cost is there to rejecting people that have questionable character?"

# TODO chunk later
# For now, creating chunks manually for first 10 paragraphs
chunks = split_text(corpus, tokenizer, 300)

# instantiate sliding window class
sliding_window = SlidingWindow(query=query, model=gpt_model)

# Call the model on each chunk
is_notepad_empty = True

print(f'Here is the query: {query}')
index = 0 # NOTE for debugging purposes
for chunk in chunks:    
    # this is the relevant information from the chunk
    print(f'This is the chunk for index {index }\n: {chunk}\n\n')
    relevant_info = sliding_window.extract_relevant_info(chunk)
    print(f'This is the relevant info for the chunk: {relevant_info} \n\n')
    
    # if it's the first iteration, our notepad just becomes the info we extracted from the first chunk
    if is_notepad_empty is True:
        print(f'Should only enter this condition if i = 0: {index}\n')
        sliding_window.notepad = relevant_info
        is_notepad_empty = False
        continue
    if 'ignore chunk' in relevant_info.choices[0].message.content.lower():
        print(f'This is the chunk I want to ignore {relevant_info.choices[0].message.content.lower()}')
        continue
    
    # synthesize the new relevant info and the notepad and use that to update running_notepad
    notepad = sliding_window.synthesis(relevant_info)
    
    # update notepad
    sliding_window.notepad = notepad # can probs consolidate w previous step
    print(f'This is the updated notepad: {sliding_window.notepad}')
    index += 1

# Give final answer
final_answer = sliding_window.final_answer()


print(f"This is the final answer: {final_answer}.")
    
