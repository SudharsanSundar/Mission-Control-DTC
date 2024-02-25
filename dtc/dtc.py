import json
from .utils import split_text
from .QAModels import GPT
from .EmbeddingModels import OpenAIEmbeddingModel

"""
Object to deal with retrieval corpus text chunks.
"""
class Corpus:
    """
    Init corpus object based on text data; automatically chunk the text.

    Params:
    - name
    - text_file: path to .txt file holding corpus data
    - chunk_size: token size of chunks for chunking process of corpus text
    - (optional) embed: whether to embed the chunks of the corpus
    """
    def __init__(self, name: str, chunk_size: int, text=None, text_file=None, embed=False):
        if not text_file and not text:
            raise Exception('Please initialize a corpus by passing in a text file path or the text itself.')
        elif text_file and text:
            raise Exception('Please initialize a corpus by passing ONE of a text file path or the text itself.')

        self.name = name
        if text_file:
            with open(text_file, "r") as f:
                self.text = f.read()
        else:
            self.text = text

        self.chunk_size = chunk_size
        self.chunks = split_text(text=self.text, max_tokens=chunk_size)
        self.num_chunks = len(self.chunks)

        if not embed:
            self.embedded_chunks = None
        else:
            self.embedded_chunks = []
            embedding_model = OpenAIEmbeddingModel()
            for chunk in self.chunks:
                self.embedded_chunks.append(embedding_model.create_embedding(chunk))


"""
Object to deal with experimental sliding window + scratchpad architecture.
"""
class MetaRNN:
    def __init__(self, qa_model: GPT, query='', verbose=False):
        """
        Init the MetaRNN with a corpus and a query.

        Parameters:
        - corpus: the dataset (our "long context")
        - query: the search query / question to answer.
        """
        self.query = query
        self.notepad = ""
        self.qa_model = qa_model
        self.extract_relevant_info_prompt = '''I am looking for information on the following: {query}. 
Here is the text: {chunk}. 
Please extract any information in the text related to this query: {query}. 
If there is no helpful or relevant information, reply simply with 'IGNORE CHUNK' and nothing else.'''

        self.synthesize_info_prompt = '''I am looking for information that helps me answer the following query: {query}. 
Please synthesize the key information in the following two paragraphs that is helpful and relevant for answering the query.
Paragraph 1: {notepad}.
Paragraph 2: {extracted_info}.
Again, please synthesize the key information in the following two paragraphs that is helpful and relevant for answering the query: {query}.'''

        self.final_answer_prompt = '''Please answer the given query based on the given information. 
Query: {query}.
Information: {notepad}. 
Again, here's the Query: {query}. 
Please answer the given query based on the given information.'''

        self.no_info_prompt = '''Please answer the given query. In your answer, please make sure to note there was no relevant information to query in the provided corpus. 
Query: {query}.'''
        self.verbose = verbose

    """
    Extract relevant info from chunk.
    
    Params:
    - chunk: str, obvious
    
    Returns:
    - answer: str, api model response
    """
    def extract_relevant_info(self, chunk: str) -> str:
        prompt = self.extract_relevant_info_prompt.format(query=self.query, chunk=chunk)
        answer = self.qa_model.answer_question(prompt)

        return answer

    """
    Synthesize info into notepad.
    
    Params:
    - extracted_info: str, obvious
    
    Returns:
    - answer: str, api model response
    """
    def synthesize_info(self, extracted_info: str) -> str:
        prompt = self.synthesize_info_prompt.format(query=self.query, notepad=self.notepad, extracted_info=extracted_info)
        self.notepad = self.qa_model.answer_question(prompt)

        return self.notepad

    """
    Runs sliding window + scratchpad technique ("MetaRNN" architecture) over the provided corpus.

    Looks at 1 chunk at a time.
    Extracts relevant info in chunk from query.
    Synthesizes extracted info with running notes.

    :param
    - corpus: Corpus object containing corpus
    - qa_model: The MetaRNN model to use for computing note states

    :returns
    - final_note_state: Final note state of MetaRNN model after reading through corpus
    - all_note_states: List of note states. Each state is a dict of the following format
        chunk : chunk of text
        relevant_info_extracted : extract of model
        synthed_note_state : note state after incorporating chunk
    """
    def get_notes_from_corpus(self, corpus: Corpus) -> (str, list):
        note_states = []

        # Read through each chunk of corpus
        for chunk in corpus.chunks:

            # Extract relevant info from the chunk, synthesize it if there's any, and then record the state
            relevant_info = self.extract_relevant_info(chunk)
            if "IGNORE CHUNK" not in relevant_info:
                synthed_info = self.synthesize_info(extracted_info=relevant_info)

                note_states.append({'chunk': chunk,
                                    'relevant_info_extracted': relevant_info,
                                    'synthed_note_state': synthed_info})
            else:
                note_states.append({'chunk': chunk,
                                    'relevant_info_extracted': relevant_info,
                                    'synthed_note_state': "SKIPPED_CHUNK"})

            if self.verbose:
                print('STATE #', len(note_states), '\n',
                      'CHUNK:', note_states[-1]['chunk'], '\n',
                      'RELEVANT INFO:', note_states[-1]['relevant_info_extracted'], '\n',
                      'NEW NOTE STATE:', note_states[-1]['synthed_note_state'], '\n',
                      '-' * 80)

        # Save all intermediate note states
        with open('all_note_states.json', 'w') as f:
            json.dump(note_states, f, indent=0, separators=(',', ': '))

        # Return final note
        return self.notepad, note_states

    """
    Answer query based on final notepad state.
    
    Params: None
    
    Returns:
    - answer: str, api model response
    """
    def final_answer(self) -> str:
        if len(self.notepad) > 0:
            prompt = self.final_answer_prompt.format(query=self.query, notepad=self.notepad)
        else:
            prompt = self.no_info_prompt.format(query=self.query)

        return self.qa_model.answer_question(prompt)



