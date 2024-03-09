from model_class import GPT, MetaRNN, Corpus, TogModel, Claude
import random
import pprint as ppr
from pprint import PrettyPrinter

pp = ppr.PrettyPrinter(indent=4)


def process_novel(txt_path):
    with open(txt_path, 'r') as f:
        file = f.read()

        file = file.replace("\n\n", "!!REPLACE!!")
        file = file.replace("\n", " ")
        file = file.replace("!!REPLACE!!", "\n\n")

    print(file)

    with open(txt_path, 'w') as f:
        f.write(file)


def process_tb(txt_path):
    with open(txt_path, 'r') as f:
        file = f.read()

        file = file.replace("\n\n", "!!REPLACE!!")
        file = file.replace("\n", " ")
        file = file.replace("!!REPLACE!!", "\n\n")

    print(file)

    with open(txt_path, 'w') as f:
        f.write(file)


def main():
    with open('dost.txt', 'r') as f:
        novel = f.read()

    with open('tb.txt', 'r') as f:
        tb = f.read()

    gpt3p5 = GPT(model='gpt-3.5-turbo-0125')
    gpt4 = GPT(model='gpt-4-0125-preview')
    mixtral54BIns = TogModel(model='mistralai/Mixtral-8x7B-Instruct-v0.1')
    claude3Sonnet = Claude()
    claude3Opus = Claude(model="claude-3-opus-20240229")

    # Add what you want


if __name__ == "__main__":
    main()
