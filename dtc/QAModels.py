
from openai import OpenAI



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
        self.client = OpenAI()

    def answer_question(self, prompt: str):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        return completion.choices[0].message.content