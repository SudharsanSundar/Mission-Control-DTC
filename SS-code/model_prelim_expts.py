from model_class import GPT, MetaRNN, Corpus
from embedding_utils import FaissIndex, OpenAIEmbeddingModel, cosine_similarity

CHUNK_SIZE200 = 200


def print_list(data: list) -> None:
    for line in data:
        print(line)


# TODO: Experiments currently inefficient: reembed chunks and states every time. fix later


"""
Compares the similarity of (query and chunk) to similarity of (query and relevant info extract from chunk).
Proof positive is that the latter is greater than the former (by a lot).

:params
- query: original query to answer
- chunks: text chunks from corpus
- note_states: all (intermediate) note states produced by the model, which includes the relevant info extract from each chunk

:returns
- similarities: list with elements of the following format
    [s1 = similarity(query, chunk_i), s2 = similarity(query, extracted_info_i), s2 - s1]
"""
def cos_sim_query_to_extract_vs_chunk(query: str, chunks: list, note_states: list) -> list:
    embedding_model = OpenAIEmbeddingModel()
    query_embd = embedding_model.create_embedding(query)
    similarities = []

    for chunk, note_state in zip(chunks, note_states):
        chunk_embd = embedding_model.create_embedding(chunk)
        extract_embd = embedding_model.create_embedding(note_state['relevant_info_extracted'])

        chunk_sim = cosine_similarity(query_embd, chunk_embd)
        extract_sim = cosine_similarity(query_embd, extract_embd)

        similarities.append([chunk_sim, extract_sim, extract_sim - chunk_sim])

    return similarities


"""
Compares the similarity of (query and chunk i) to similarity of (query and synthesized note i).
Proof positive is that the latter is greater than the former (by a lot).

:params
- query: original query to answer
- chunks: text chunks from corpus
- note_states: all (intermediate) note states produced by the model, which includes the note state at each chunk

:returns
- similarities: list with elements of the following format
    [s1 = similarity(query, chunk_i), s2 = similarity(query, note_state_i), s2 - s1]
"""
def cos_sim_query_to_note_vs_chunk(query: str, chunks: list, note_states: list) -> list:
    embedding_model = OpenAIEmbeddingModel()
    query_embd = embedding_model.create_embedding(query)
    similarities = []

    for chunk, note_state in zip(chunks, note_states):
        chunk_embd = embedding_model.create_embedding(chunk)
        note_embd = embedding_model.create_embedding(note_state['synthed_note_state'])

        chunk_sim = cosine_similarity(query_embd, chunk_embd)
        note_sim = cosine_similarity(query_embd, note_embd)

        similarities.append([chunk_sim, note_sim, note_sim - chunk_sim])

    return similarities


"""
Shows the similarity of (query and note state i) and similarity of (query and relevant info extract) over time.
Proof positive is that the former increases over time, generally in proportion to the latter.

:params
- query: original query to answer
- note_states: all (intermediate) note states produced by the model, which includes the note state and info extracted at each chunk

:returns
- similarities: list with elements of the following format
    [s1 = similarity(query, note_state_i), s2 = similarity(query, relevant_info_i)]
"""
def cos_sim_query_to_note_and_extract_over_time(query: str, note_states: list) -> list:
    embedding_model = OpenAIEmbeddingModel()
    query_embd = embedding_model.create_embedding(query)
    similarities = []

    for note_state in note_states:
        extract_embd = embedding_model.create_embedding(note_state['relevant_info_extracted'])
        note_embd = embedding_model.create_embedding(note_state['synthed_note_state'])

        extract_sim = cosine_similarity(query_embd, extract_embd)
        note_sim = cosine_similarity(query_embd, note_embd)

        similarities.append([note_sim, extract_sim])

    return similarities


def main():
    # # # Set up pipeline
    # Model used for API calls
    gpt3p5 = GPT()

    # Corpus used for note-taking
    corpus_path = r'test_corpus.txt'
    dud_corpus_path = r'dud_corpus.txt'

    corpus = Corpus(name='The Cask of Amontillado, Edgar Allan Poe', text_file=corpus_path, chunk_size=CHUNK_SIZE200)
    # corpus = Corpus(name='On the Origin of Species, Charles Darwin', text_file=dud_corpus_path, chunk_size=CHUNK_SIZE200)

    query = "What was the reason that incited the narrator to kill his so-called friend?"

    # MetaRNN model for note-taking and synthesis
    qa_model = MetaRNN(query=query, model=gpt3p5)

    # # # Get final notes
    print('-' * 80)
    final_notes, note_states = qa_model.get_notes_from_corpus(corpus)

    # # # Answer query
    final_answer = qa_model.final_answer()

    # # # Run experiments
    exp1 = cos_sim_query_to_extract_vs_chunk(query, corpus.chunks, note_states)
    exp2 = cos_sim_query_to_note_vs_chunk(query, corpus.chunks, note_states)
    exp3 = cos_sim_query_to_note_and_extract_over_time(query, note_states)

    # # # Display results
    print_list(exp1)
    print('-'*80)
    print_list(exp2)
    print('-'*80)
    print_list(exp3)


if __name__ == "__main__":
    main()
