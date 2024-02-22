from openai import OpenAI
from chunk_utils import split_text

# TODO: Delete before push!!!
SS_API_KEY = ''
client = OpenAI(api_key=SS_API_KEY)


"""
Object to deal with API calls to OAI models.
"""
class GPT:
    """
    Params
    - model | model to make api call to
    - system_prompt | system_prompt to use for model
    """
    def __init__(self,
                 model="gpt-3.5-turbo",
                 system_prompt="You are a Question Answering portal."):
        self.model = model
        self.system_prompt = system_prompt

    def answer(self, prompt: str):
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        return completion

    def answer_txt(self, prompt: str) -> str:
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        return completion.choices[0].message.content


"""
Object to deal with experimental sliding window + scratchpad architecture.
"""
class MetaRNN:
    def __init__(self, query: str, model: GPT):
        """
        Init the MetaRNN with a corpus and a query.

        Parameters:
        - corpus: the dataset (our "long context")
        - query: the search query / question to answer.
        """
        self.query = query
        self.notepad = ""
        self.model = model

    """
    Extract relevant info from chunk.
    
    Params:
    - chunk: str, obvious
    
    Returns:
    - answer: str, api model response
    """
    def extract_relevant_info(self, chunk: str) -> str:
        prompt = f'''I am looking for information on the following: 
        {self.query}. Here is the text: {chunk}. Please extract any information in the text related 
        to this query: {self.query}. If there is no helpful or relevant information, 
        reply simply with 'IGNORE CHUNK' and nothing else. '''

        answer = self.model.answer_txt(prompt)

        return answer

    """
    Synthesize info into notepad.
    
    Params:
    - extracted_info: str, obvious
    
    Returns:
    - answer: str, api model response
    """
    def synthesize_info(self, extracted_info: str) -> str:
        prompt = f'''I am looking for information that helps me answer the following query: {self.query}. 
        Please synthesize the key information in the following two paragraphs that is helpful and relevant for answering the query.
        Paragraph 1: {self.notepad}.
        Paragraph 2: {extracted_info}.
        Again, please synthesize the key information in the following two paragraphs that is helpful and relevant for answering the query: {self.query}.'''

        self.notepad = self.model.answer_txt(prompt)

        return self.notepad

    """
    Answer query based on final notepad state.
    
    Params: None
    
    Returns:
    - answer: str, api model response
    """
    def final_answer(self) -> str:
        if len(self.notepad) > 0:
            prompt = f'''Please answer the given query based on the given information. 
            Information: {self.notepad}. 
            Query: {self.query}. 
            Again, please answer the given query based on the given information.'''
        else:
            prompt = f'''Please answer the given query. In your answer, please note there was no relevant information to query in the provided corpus. 
                        Query: {self.query}. 
                        Again, please answer the given query based on the given information. Again, please make sure to note there was no relevant information to query in the provided corpus.'''

        return self.model.answer_txt(prompt)


"""
Object to deal with retrieval corpuses.
"""
class Corpus:
    """
    Init corpus object based on text data; automatically chunk the text.

    Params:
    - name
    - text_file: path to .txt file holding corpus data
    - chunk_size: token size of chunks for chunking process of corpus text
    """
    def __init__(self, name: str, text_file: str, chunk_size: int):
        self.name = name
        with open(text_file, "r") as f:
            self.text = f.read()
        self.chunk_size = chunk_size
        self.chunks = split_text(text=self.text, max_tokens=chunk_size)
        self.num_chunks = len(self.chunks)
