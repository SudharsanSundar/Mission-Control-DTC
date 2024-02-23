from model_class import GPT, MetaRNN, Corpus

CHUNK_SIZE200 = 200


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
    final_notes, _ = qa_model.get_notes_from_corpus(corpus)

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
