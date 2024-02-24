from model_class import GPT, MetaRNN, Corpus
from embedding_utils import FaissIndex, OpenAIEmbeddingModel, cosine_similarity
from eval_data_utils import load_quality_data
import json

CHUNK_SIZE200 = 200


# TODO: Salman: get this working with ur quality script and run the eval

def eval_naive_model(dataset: list, model) -> list:
    results = []

    for ex in dataset:
        # TODO: format prompt
        prompt = ''

        model_answer = model.answer_txt(prompt)

        results.append({'question': prompt,
                        'model answer': model_answer,
                        'correct answer': ex.answer})

    return results


def eval_exp_model(dataset: list, model):
    results = []

    for ex in dataset:
        # TODO: create corpus for retrieval model
        corpus = Corpus(name='', chunk_size=None, text='')
        # TODO: create query for retrieval model (answer choices, etc.)
        model.query = ex.question

        final_note, note_states = model.get_info_from_corpus(corpus)
        model_answer = model.final_answer()

        results.append({'question': model.query,
                        'model answer': model_answer,
                        'note states': note_states,
                        'correct answer': ex.answer})

    return results


def main():
    # # # Set up models
    gpt3p5 = GPT()
    qa_model = MetaRNN(model=gpt3p5)

    gpt3p5Ctx = GPT(model="gpt-3.5-turbo-0125") # supports 16k context

    # # # Set up eval data
    eval_data_path = '../data/quality.train'    # NOTE: this is the 1.0.1 version, HTML stripped
    quality_dataset = load_quality_data(eval_data_path)

    # NOTE: slice dataset to be smaller

    # # # Eval models
    naive_results = eval_naive_model(quality_dataset, gpt3p5Ctx)
    exp_results = eval_exp_model(quality_dataset, qa_model)

    # # # Save eval results
    with open('naive_gpt3p5-0125_quality_eval.json', 'w') as f:
        json.dump(naive_results, f, indent=0, separators=(',', ': '))

    with open('metarnn_quality_eval.json', 'w') as f:
        json.dump(exp_results, f, indent=0, separators=(',', ': '))


if __name__ == "__main__":
    main()