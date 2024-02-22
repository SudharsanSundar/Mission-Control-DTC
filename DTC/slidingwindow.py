from openai import OpenAI
import os


client = OpenAI()


class GPT:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
    
    def answer(self, prompt):
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a Question Answering portal."},
                {"role": "user", "content": prompt}
            ] 
        )
        return completion


class SlidingWindow:
    def __init__(self, query, model):
        """
        Initialize the SlidingWindow with a corpus and a query.

        Parameters:
        - corpus: the dataset (our "long context")
        - query: the search query / question to answer.
        """
        #self.corpus = corpus ### TODO rename corpus to chunk later 
        self.query = query
        self.notepad = ""  # Initialize the answer notepad
        self.model = model

    """
    We want to extract the information from the chunk that is most relevant
    to the query. 
    
    Inputs: query, chunk (of text)
    Output: extracted string of relevant information
    """
    def extract_relevant_info(self, chunk):
        prompt = f'''I am looking for information on the following: 
        {self.query}. Here is the text: {chunk}. Please extract any information in the text related 
        to the query, which was: {self.query}. If there is no helpful or relevant information, 
        reply simply with 'IGNORE CHUNK' and nothing else. '''
        answer = self.model.answer(prompt)
        return answer
    
    """
    We want to synthesize our notepad and the relevant information that was just
    extracted from the chunk, in order to generate our new notepad. 

    Inputs: 
    - notepad: current note pad 
    - current_output: the output from extract_relevant_info generated from the current chunk 
    Output: 
    - newly generated notepad, generated from synthesizing the notepad and our current_output
    """
    def synthesis(self, extracted_info):
        prompt = f'''I have two paragraphs that cover information about the following: {self.query}. 
        I am looking for a synthesis of the information in both. 
        Paragraph 1: {self.notepad}. Paragraph 2: {extracted_info}. 
        Please  synthesize the key information from these paragraphs as it pertains to the query, which was:
        {self.query}.'''
        self.notepad = self.model.answer(prompt)
        return self.notepad
        
    """
    We want to give a final answer to the query using just the notes we've taken.
    Inputs: 
    - query
    - current notepad
    Output: 
    - final_answer
    """
    def final_answer(self):
        prompt = f'''Based on the following summary, I have a question. Summary: {self.notepad}. 
        Given this summary, my question is {self.query}. Could you answer this based on the summarized information? 
        Here is the query again for reference: {self.query}'''
        self.notepad = self.model.answer(prompt)
        return self.notepad


            
