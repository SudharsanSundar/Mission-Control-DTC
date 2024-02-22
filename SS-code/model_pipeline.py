from model_class import GPT, MetaRNN, Corpus
import json

CHUNK_SIZE200 = 200

"""
Runs sliding window + scratchpad technique ("MetaRNN" architecture) over the provided corpus.

Looks at 1 (mutually exclusive, sequential) chunk at a time.
Extract relevant info in chunk from query.
Synthesizes extracted info with running notes.

:param
- corpus: Corpus object containing corpus
- qa_model: The MetaRNN model to use for computing note states

:returns
- final_note_state: Final note state of MetaRNN model after reading through corpus
- 

"""
def get_notes_from_corpus(corpus: Corpus, qa_model: MetaRNN) -> str:
    note_states = []

    # Read through each chunk of corpus
    for chunk in corpus.chunks:

        # Extract relevant info from the chunk, synthesize it if there's any, and then record the state
        relevant_info = qa_model.extract_relevant_info(chunk)
        if "IGNORE CHUNK" not in relevant_info:
            synthed_info = qa_model.synthesize_info(extracted_info=relevant_info)

            note_states.append({'chunk': chunk,
                                'relevant_info_extracted': relevant_info,
                                'synthed_note_state': synthed_info})
        else:
            note_states.append({'chunk': chunk,
                                'relevant_info_extracted': relevant_info,
                                'synthed_note_state': "SKIPPED_CHUNK"})

        print('STATE #', len(note_states), '\n',
              'CHUNK:', note_states[-1]['chunk'], '\n',
              'RELEVANT INFO:', note_states[-1]['relevant_info_extracted'], '\n',
              'NEW NOTE STATE:', note_states[-1]['synthed_note_state'], '\n',
              '-' * 80)

    # Save all intermediate note states
    with open('all_note_states.json', 'w') as f:
        json.dump(note_states, f, indent=0, separators=(',', ': '))

    return qa_model.notepad if len(qa_model.notepad) > 0 else repr(qa_model.notepad)


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
    final_notes = get_notes_from_corpus(corpus, qa_model)

    # # # Answer query
    final_answer = qa_model.final_answer()

    # # # Display results
    print("TASK:\n", "Query:", query, "\nCorpus:", corpus.name)
    print("-" * 20)
    print("Final notes:\n\n", final_notes)
    print("-" * 20)
    print("Final answer:\n\n", final_answer)


if __name__ == "__main__":
    main()
