from openai import OpenAI
import json
import anthropic
# import google
# import google.cloud as google_ai_platform
# from vertexai.preview.generative_models import GenerativeModel
# import pathlib
# import textwrap
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

# TODO: Delete before push!!! Add your own API key!!!


client = OpenAI(api_key=OAI_API_KEY)
togClient = OpenAI(api_key=TOG_API_KEY, base_url='https://api.together.xyz')
antClient = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
# google_gen_ai.configure(api_key=GOOGLE_API_KEY)


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
                 model="gpt-3.5-turbo-0125",
                 system_prompt="You are a helpful assistant."):
        self.model = model
        self.system_prompt = system_prompt

    def answer_txt(self, prompt: str) -> str:
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        return completion.choices[0].message.content


"""
Handles calls to tgoether ai models
"""
class TogModel:
    def __init__(self,
                 model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                 system_prompt="You are a helpful assistant.",
                 max_tokens=1024):
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def answer_txt(self, prompt: str) -> str:
        completion = togClient.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        return completion.choices[0].message.content


"""
Handles calls to anthropic models
"""
class Claude:
    def __init__(self,
                 model="claude-3-sonnet-20240229",
                 system_prompt="You are a helpful assistant.",
                 max_tokens=2048):
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

    def answer_txt(self, prompt: str) -> str:
        response = antClient.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.0,
            system=self.system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.content[0].text


"""
Handles calls to google deepmind models
"""
class Gemini:
    def __init__(self,
                 model="gemini-1.5-pro",
                 system_prompt="You are a helpful assistant.",
                 max_tokens=2048):
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

    def answer_txt(self, prompt: str) -> str:
        # model = GenerativeModel(self.model)
        # response = model.generate_content(prompt)
        #
        # return response.candidates[0].content.parts[0].text
        return 'not implemented'
