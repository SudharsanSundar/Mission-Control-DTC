from model_class import GPT, MetaRNN, Corpus
from embedding_utils import FaissIndex, OpenAIEmbeddingModel, cosine_similarity
import matplotlib.pyplot as plt



CHUNK_SIZE200 = 200


"""
Print utility
"""
def print_list(data: list) -> None:
    for line in data:
        print(line)


"""
Gets the cosine similarity of (query and chunk i), (query and relevant info extract from chunk i), (query, note state i after synthesizing info)
Proof positive is that the last two are greater than the first, and that the last one grows over time/in proportion to the second

:params
- query_embd: original query to answer, embedding
- note_state_embds: list of dicts with each dict containing 
    {
    'chunk': chunk embedding,
    'relevant_info_extracted': extract embedding,
    'synthed_note_state: note state embedding
    }

:returns
- query_to_chunk: cos sim from query to each chunk
- query_to_extract: cos sim from query to each extract
- query_to_note: cos sim from query to each note state
"""
def cos_sim_query_to_chunk_extract_note(query_embd, note_state_embds: list) -> (list, list, list):
    query_to_chunk = []
    query_to_extract = []
    query_to_note = []

    for note_state in note_state_embds:
        chunk_sim = cosine_similarity(query_embd, note_state['chunk'])
        extract_sim = cosine_similarity(query_embd, note_state['relevant_info_extracted'])
        note_sim = cosine_similarity(query_embd, note_state['synthed_note_state'])

        query_to_chunk.append(chunk_sim)
        query_to_extract.append(extract_sim)
        query_to_note.append(note_sim)

    return query_to_chunk, query_to_extract, query_to_note


"""
Embeds all info in the note_states return object from calling .get_notes_from_corpus()
"""
def create_note_state_dict_embeddings(note_states: list) -> list:
    note_state_embds = []
    embedding_model = OpenAIEmbeddingModel()

    for item in note_states:
        note_state_embds.append({'chunk': embedding_model.create_embedding(item['chunk']),
                                 'relevant_info_extracted': embedding_model.create_embedding(item['relevant_info_extracted']),
                                 'synthed_note_state': embedding_model.create_embedding(item['synthed_note_state'])
        })

    return note_state_embds

def run_model(corpus, query, model):
    # # # Set up pipeline
    # Model used for API calls
    

    # Corpus used for note-taking
    #corpus_path = r'test_corpus.txt'
    #corpus_path = r'Iranian Revolution Galvin Chapter.txt'
    #dud_corpus_path = r'dud_corpus.txt'

    #corpus = Corpus(name='The Iranian Revolution', text_file=corpus_path, chunk_size=CHUNK_SIZE200)
    #corpus = Corpus(name='The Cask of Amontillado, Edgar Allan Poe', text_file=corpus_path, chunk_size=CHUNK_SIZE200)
    # corpus = Corpus(name='On the Origin of Species, Charles Darwin', text_file=dud_corpus_path, chunk_size=CHUNK_SIZE200)

    #query = "What was the reason that incited the narrator to kill his so-called friend?"
    #query = "What justification do social scientists provide for the fact that the Iranian Revolution occured?"


    # MetaRNN model for note-taking and synthesis
    qa_model = MetaRNN(query=query, model=model)

    # # # Get final notes
    print('-' * 80)
    final_notes, note_states = qa_model.get_notes_from_corpus(corpus)

    # # # Answer query
    final_answer = qa_model.final_answer()

    # # # Run experiments
    embedding_model = OpenAIEmbeddingModel()
    query_embd = embedding_model.create_embedding(query)
    note_state_dict_embds = create_note_state_dict_embeddings(note_states)

    query_to_chunk, query_to_extract, query_to_note = cos_sim_query_to_chunk_extract_note(query_embd, note_state_dict_embds)
    return query_to_chunk, query_to_extract, query_to_note

def plot(data):
    plt.figure(figsize=(14, 7))
    plt.plot(data, marker='o', linestyle='-', color='b')
    plt.title('[Title] Similarity')
    plt.xlabel('Chunk Number')
    plt.ylabel('Cosine Similarity')
    plt.xticks(range(len(data)))  # Ensure we have a tick for each chunk
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Show plot
    plt.show()

def main():
    # # # Set up pipeline
    # Model used for API calls
    
    model = GPT()
    # Corpus used for note-taking
    #corpus_path = r'test_corpus.txt'
    #corpus_path = r'Iranian Revolution Galvin Chapter.txt'
    dud_corpus_path = r'dud_corpus.txt'

    #corpus = Corpus(name='The Iranian Revolution', text_file=corpus_path, chunk_size=CHUNK_SIZE200)
    #query = "What justification do social scientists provide for the fact that the Iranian Revolution occured?"

    # query without answer provided in corpus - distractor
    #query = "What is Sudharsan's favorite color?"

    # query with random answer inserted (easter egg)
    #query = "What is Toyon?"

    #query = "How does the narrator keep his friend from turning back?" ### TODO RELEVANT FOR Amontillado
    query = "What is Toyon Hall?"

    corpus_path = r'test_corpus.txt'
    corpus = Corpus(name='The Cask of Amontillado', text_file=corpus_path, chunk_size=CHUNK_SIZE200)
    query_to_chunk, query_to_extract, query_to_note = run_model(corpus, query, model)


    # # # Display results
    # TODO: plot results, analyze them
    #print(f'this is query_to_chunk: {query_to_chunk}')
    #print(f'this is query_to_extract: {query_to_extract}')
    #print(f'this is query_to_note: {query_to_note}')

    # plot query_to_chunk

    # Create figure and plot
    
    ### THIS is QUERY TO CHUNK COSINE SIMILARITY
    """
    plt.figure(figsize=(14, 7))
    plt.plot(query_to_chunk, marker='o', linestyle='-', color='b')
    plt.title('Query to Chunk Cosine Similarity')
    plt.xlabel('Chunk Number')
    plt.ylabel('Cosine Similarity')
    plt.xticks(range(len(query_to_chunk)))  # Ensure we have a tick for each chunk
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Show plot
    plt.show()
    """

    ### THIS is QUERY TO CHUNK COSINE SIMILARITY
    """
    plt.figure(figsize=(14, 7))
    plt.plot(query_to_extract, marker='o', linestyle='-', color='b')
    plt.title('Query to Extract Cosine Similarity')
    plt.xlabel('Chunk Number')
    plt.ylabel('Cosine Similarity')
    plt.xticks(range(len(query_to_extract)))  # Ensure we have a tick for each chunk
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    

    # Show plot
    plt.show()
    """
    # Create the plot
    plt.figure(figsize=(14, 7))

    # Plot each series
    plt.plot(query_to_chunk, marker='o', linestyle='-', color='blue', label='Query to Chunk')
    plt.plot(query_to_extract, marker='x', linestyle='--', color='red', label='Query to Extract')
    plt.plot(query_to_note, marker='^', linestyle='-.', color='green', label='Query to Note')

    # Adding plot title and labels
    plt.title('Cosine Similarity Measures for Query Comparisons')
    plt.xlabel('Chunk Number')
    plt.ylabel('Cosine Similarity')

    # Adding a legend to explain each line
    plt.legend()

    # Displaying the grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Show the plot
    plt.tight_layout()
    plt.show()







    # TODO: use better corpuses, etc.


if __name__ == "__main__":
    main()
