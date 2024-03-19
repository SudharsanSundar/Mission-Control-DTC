from model_class import GPT, TogModel, Claude
from io_processing import stream_jsonl, write_jsonl, list_generator, jsonl_to_list
import random
import pprint as ppr
import tiktoken
import os
import time
import tqdm

pp = ppr.PrettyPrinter(indent=4)
gpt_encoding = tiktoken.get_encoding('cl100k_base')


# Gets first n tokens of numerical haystack data (AN INTRODUCTION TO MATHEMATICS By A. N. WHITEHEAD)
def get_numerical_hay(tokens: int, fp: str = './hay_data/numerical.txt') -> (str, int):
    with open(fp, 'r') as f:
        corpus = f.read()

    enc_corpus = gpt_encoding.encode(corpus)
    enc_corpus_trim = enc_corpus[:tokens]
    corpus_trim = gpt_encoding.decode(enc_corpus_trim)
    total_tok = len(gpt_encoding.encode(corpus_trim))

    print('HAY:', total_tok, 'tokens of NUMERICAL data')

    return corpus_trim, total_tok


# Gets approx first n tokens of code haystack data (concatenated HumanEval function solutions)
def get_code_hay(tokens: int, fp: str = './hay_data/code.txt') -> (str, int):
    with open(fp, 'r') as f:
        corpus = f.read()

    # Get all list of all HumanEval functions
    funcs = corpus.split('- / - / - /')

    # Keep packing in functions until we hit desired threshold
    corpus_trim = ''
    for func in funcs:
        corpus_trim += func + '\n\n- / -\n\n'
        if len(gpt_encoding.encode(corpus_trim)) > tokens:
            break

    total_tok = len(gpt_encoding.encode(corpus_trim))

    print('HAY:', total_tok, 'tokens of CODE data')

    return corpus_trim, total_tok


# Solves for objective correct answer to needle tasks
def rec_solve(vals):
    if len(vals) == 1:
        return vals[0][0]
    else:
        cur = vals[0]
        ans = rec_solve(vals[1:])
        if cur[1] == ' plus ':
            return ans + cur[0]
        elif cur[1] == ' minus ':
            return ans - cur[0]
        else:
            print('somethings wrong with rec solve...')


# Create essence of needles used
def create_needle_vals(num_needles):
    operations = [' plus ', ' minus ']
    min = 0
    max = 10
    needle_cores = []
    distractor_cores = []

    # Create each needle's core, i.e. what operation it performs and what random number it uses
    for i in range(num_needles):
        value = random.randint(min, max)
        op = random.choice(operations)

        if i < num_needles - 1:
            needle_cores.append([value, op])
        else:
            needle_cores.append([value, None])

    # Create cores for distractors
    for i in range(num_needles):
        value = random.randint(min, max)
        op = random.choice(operations)

        if i < num_needles - 1:
            distractor_cores.append([value, op])
        else:
            distractor_cores.append([value, None])

    correct_ans = rec_solve(needle_cores)

    return needle_cores, distractor_cores, correct_ans


# Create sentences for each numerical needle
def create_numerical_needles(num_needles, hard_mode=False):
    needle_cores, distractor_cores, correct_ans = create_needle_vals(num_needles)
    needles = []
    distractors = []

    if not hard_mode:
        # E.g. needle 3 -> needle 2 -> needle 1 -> needle 0
        for i, core in zip(range(len(needle_cores)), needle_cores):
            if i < len(needle_cores) - 1:
                needle = f'The value of Needle {i} is equal to the value of Needle {i+1}{core[1]}{core[0]}.'
            else:
                needle = f'The value of Needle {i} is equal to {core[0]}.'

            needles.append(needle)
    elif hard_mode:
        # E.g. needle 1 -> needle 5 -> needle 3; irrelevantly, needle 0 -> needle 6 -> needle 4
        shuffled_ids = random.shuffle([i for i in range(len(needle_cores) * 2)])

        for i in range(len(needle_cores)):
            id = shuffled_ids[i]
            core = needle_cores[i]

            if i < len(needle_cores) - 1:
                next_id = shuffled_ids[i+1]
                needle = f'The value of Needle {id} is equal to the value of Needle {next_id}{core[1]}{core[0]}.'
            else:
                needle = f'The value of Needle {id} is equal to {core[0]}.'

            needles.append(needle)

        for i in range(len(distractor_cores)):
            id = shuffled_ids[i + len(needle_cores)]
            core = distractor_cores[i + len(needle_cores)]

            if i < len(distractor_cores) - 1:
                next_id = shuffled_ids[i + 1 + len(needle_cores)]
                distractor = f'The value of Needle {id} is equal to the value of Needle {next_id}{core[1]}{core[0]}.'
            else:
                distractor = f'The value of Needle {id} is equal to {core[0]}.'

            distractors.append(distractor)

    shuffled_needles = needles.copy()
    random.shuffle(shuffled_needles)

    return needles, shuffled_needles, correct_ans, needle_cores, distractors


# Helper for creating function format of needle
def symbolize_core(core):
    new_core = []

    new_core.append(core[0])

    if core[1] == ' plus ':
        new_core.append('+')
    elif core[1] == ' minus ':
        new_core.append('-')
    else:
        new_core.append(None)

    return new_core


# Create function for each code needle
def create_code_needles(num_needles, hard_mode=False):
    needle_cores, distractor_cores, correct_ans = create_needle_vals(num_needles)
    needles = []
    distractors = []

    if not hard_mode:
        # E.g. needle 3 -> needle 2 -> needle 1 -> needle 0
        for i, core in zip(range(len(needle_cores)), needle_cores):
            symbol_core = symbolize_core(core)
            if i < len(needle_cores) - 1:
                needle = f'\n\ndef get_value_of_needle_{i}():\n    return get_value_of_needle_{i + 1}() {symbol_core[1]} {core[0]}\n\n'
            else:
                needle = f'\n\ndef get_value_of_needle_{i}():\n    return {core[0]}\n\n'

            needles.append(needle)
    elif hard_mode:
        # E.g. needle 1 -> needle 5 -> needle 3; irrelevantly, needle 0 -> needle 6 -> needle 4
        raise Exception('not implemented!!')

    shuffled_needles = needles.copy()
    random.shuffle(shuffled_needles)

    return needles, shuffled_needles, correct_ans, needle_cores, distractors


# Create full numerical prompt
def create_numerical_prompt(needles_used, hay=None):
    if hay:
        hay_atoms = hay.split(' ')
        num_needles = len(needles_used)

        rand_pos = random.sample(range(0, len(hay_atoms) - 1), num_needles)
        rand_pos.sort()

        for needle, pos in zip(needles_used, rand_pos):
            needle_fmt = f'({needle})'
            hay_atoms.insert(pos, needle_fmt)

        information = ' '.join(hay_atoms)
    else:
        information = '\n'.join(needles_used)

    preamble = "Please use the following information to correctly answer the following query.\n\n"
    reminder = 'Again, please use the above information to correctly answer the following query: "'
    query = 'What is the value of Needle 0?\n\n'
    engr = "Let's think step by step."

    prompt = f'{preamble}Query: {query}Information:\n{information}\n\n{reminder}{query}{engr}'

    return prompt


# Create full code prompt
def create_code_prompt(needles_used, hay=None):
    if hay:
        hay_atoms = hay.split('\n\n- / -\n\n')
        num_needles = len(needles_used)

        rand_pos = random.sample(range(0, len(hay_atoms) - 1), num_needles)
        rand_pos.sort()

        for needle, pos in zip(needles_used, rand_pos):
            needle_fmt = f'({needle})'
            hay_atoms.insert(pos, needle_fmt)

        information = ' '.join(hay_atoms)
    else:
        information = ''.join(needles_used)

    preamble = "Please use the following information to correctly answer the following query.\n\n"
    reminder = 'Again, please use the above information to correctly answer the following query: '
    query = 'What is the value returned when calling \'get_value_of_needle_0()\'?\n\n'
    engr = "Let's think step by step."

    prompt = f'{preamble}Query: {query}Information:\n{information}\n\n{reminder}{query}{engr}'

    return prompt


# Used to automatically get int value of answer from model response.
# Relies on trend that it's typically the last word of its response
def extract_num(ans):
    words = ans.split(' ')
    final_word = words[-1]
    cleaned_num = ''

    for char in final_word:
        if char.isdigit() or char == '-':
            cleaned_num += char
    try:
        return int(cleaned_num)
    except ValueError:
        return 'Last word not numerical value: ' + final_word


def create_numerical_hay_dict(high_end_exclusive: int):
    hay_dict = {}
    for i in range(1, high_end_exclusive, 1):
        key = f'hay{i}k'
        hay_dict[key] = get_numerical_hay(i * 1000)

    return hay_dict


def create_code_hay_dict(high_end_exclusive: int):
    hay_dict = {}
    for i in range(1, high_end_exclusive, 1):
        key = f'hay{i}k'
        hay_dict[key] = get_code_hay(i * 1000)

    return hay_dict


# Creates a desired eval dataset (prompt, correct ans num)
def create_progressive_needles_eval_data(num_needles: int,
                                        needle_func,
                                        num_qs: int,
                                        q_type: str,
                                        out_fp: str,
                                        hay_max_exclusive: int,
                                        seed=42) -> list:
    random.seed(seed)

    if q_type == 'numerical':
        hay_dict = create_numerical_hay_dict(hay_max_exclusive)
    elif q_type == 'code':
        hay_dict = create_code_hay_dict(hay_max_exclusive)

    eval_data = []

    for i in tqdm.tqdm(range(num_qs)):
        hay, hay_tok = None, 0
        eval_question = {}
        hay_tokens = {'needles-only-prompt': 0}

        needles, shuffled_needles, ans, cores, _ = needle_func(num_needles)  # not using needle cdistractors
        eval_question['needles-cores'] = cores
        eval_question['correct-ans'] = ans

        if q_type == 'numerical':
            prompt = create_numerical_prompt(needles_used=shuffled_needles, hay=hay)
        elif q_type == 'code':
            prompt = create_code_prompt(needles_used=shuffled_needles, hay=hay)

        eval_question['needles-only-prompt'] = prompt

        for hay_amount in hay_dict:
            hay, hay_tok = hay_dict[hay_amount]

            if q_type == 'numerical':
                prompt = create_numerical_prompt(needles_used=shuffled_needles, hay=hay)
            elif q_type == 'code':
                prompt = create_code_prompt(needles_used=shuffled_needles, hay=hay)

            eval_question[hay_amount + '-prompt'] = prompt
            hay_tokens[hay_amount] = hay_tok

        eval_question['hay-tokens'] = hay_tokens

        eval_data.append(eval_question)

        # Save eval results
        write_jsonl(out_fp, list_generator(eval_data))

    print('Created eval data, saved to', out_fp)

    return eval_data


# Step by step answer, supposed to be correct response for finetuning
def generate_correct_model_numerical_answer(needles, shuffled_needles, needle_cores, correct_ans):
    relevant_info = '\n'.join([f'{i+1}. {needle}' for i, needle in zip(range(len(shuffled_needles)), shuffled_needles)])
    ordered_info = '\n'.join([f'{i+1}. {needle}' for i, needle in zip(range(len(needles)), reversed(needles))])

    solution_steps = ''

    running_sum = 0
    for i, core in zip(range(len(needle_cores)), reversed(needle_cores)):
        running_sum_new = running_sum
        core = symbolize_core(core)

        if core[1] != '-':
            running_sum_new += core[0]
        else:
            running_sum_new -= core[0]

        if i == 0:
            step = f'{i+1}. The value of Needle {len(needle_cores) - i - 1} is {core[0]}\n'
        else:
            step = f'{i + 1}. The value of Needle {len(needle_cores) - i - 1} is the value of Needle ' \
                   f'{len(needle_cores) - i} {core[1]} {core[0]}. Hence, the value of Needle ' \
                   f'{len(needle_cores) - i - 1} is equal to {running_sum} {core[1]} {core[0]} = {running_sum_new}.\n'

        solution_steps += step
        running_sum = running_sum_new

    assert running_sum == correct_ans

    answer = f'''\nAnswering this query involves combining facts about the values of various Needles, since the value of Needle 0 is dependent on the value of other Needles. Thus, let's first gather all of the relevant information needed to answer this query. 

Based on the information provided, the relevant information for finding the value of Needle 0 can be summarized as follows:
{relevant_info}

Reordering this information to make it easier for us to compute the value of Needle 0, we have the following:
{ordered_info}

Now, let's solve for the value of Needle 0 step by step:
{solution_steps}
Thus, the value of Needle 0 is {correct_ans}.'''

    return answer


# Step by step answer, supposed to be correct response for finetuning
def generate_correct_model_code_answer(needles, shuffled_needles, needle_cores, correct_ans):
    relevant_info = '\n'.join([f'{i + 1}. \'\n{needle[2:-2]}\n\'' for i, needle in zip(range(len(shuffled_needles)), shuffled_needles)])
    ordered_info = '\n'.join([f'{i + 1}. \'\n{needle[2:-2]}\n\'' for i, needle in zip(range(len(needles)), reversed(needles))])

    solution_steps = ''

    running_sum = 0
    for i, core, needle in zip(range(len(needle_cores)), reversed(needle_cores), reversed(needles)):
        running_sum_new = running_sum
        core = symbolize_core(core)
        needle = '\n\'' + needle[2:-2] + '\n\''

        if core[1] != '-':
            running_sum_new += core[0]
        else:
            running_sum_new -= core[0]

        if i == 0:
            step = f'{i + 1}. {needle}\nHence, the return value of \'def get_value_of_needle_{len(needle_cores) - i - 1}()\' ' \
                   f'is {running_sum_new}.\n'
        else:
            step = f'{i + 1}. {needle}\nHence, the return value of \'def get_value_of_needle_{len(needle_cores) - i - 1}()\' ' \
                   f'is equal to {running_sum} {core[1]} {core[0]} = {running_sum_new}.\n'

        solution_steps += step
        running_sum = running_sum_new

    assert running_sum == correct_ans

    answer = f'''\nAnswering this query involves correctly stepping through chained function calls, since the return value of \'get_value_of_needle_0()\' is dependent on the return value of other functions. Thus, let's first gather all of the relevant information needed to answer this query.

Based on the information provided, the relevant information for finding the return value of \'get_value_of_needle_0()\' can be summarized as follows:
{relevant_info}

Reordering this information to make it easier for us to compute the return value of \'get_value_of_needle_0()\', we have the following:
{ordered_info}

Now, let's solve for the return value of \'get_value_of_needle_0()\' step by step:
{solution_steps}
Thus, the return value of \'get_value_of_needle_0()\' is {correct_ans}.'''

    return answer


# Creates a desired eval dataset of form (prompt, correct ans num). Can customize to your liking.
def create_progressive_needles_finetuning_data(num_needles_max: int,
                                             needle_func,
                                             num_qs: int,
                                             q_type: str,
                                             out_fp: str,
                                             num_tokens_hay_max: int,
                                             seed=42**2) -> list:
    random.seed(seed)
    hay = None

    if num_tokens_hay_max < 1000:
        raise Exception('max num hay tokens too small! make it >= 1000')

    if num_needles_max > 10:
        print('Warning: large num max needles might lead to error when trying to make haystack prompt')
    elif num_needles_max < 2:
        raise Exception('too few max num needles! must be >= 2')

    ft_data = []

    for i in tqdm.tqdm(range(num_qs)):
        num_tokens_hay = random.randint(999, num_tokens_hay_max)
        num_needles = random.randint(2, num_needles_max)

        if q_type == 'numerical':
            hay, _ = get_numerical_hay(num_tokens_hay)
        elif q_type == 'code':
            hay, _ = get_code_hay(num_tokens_hay)

        needles, shuffled_needles, ans, needle_cores, _ = needle_func(num_needles)  # not using distractors

        if q_type == 'numerical':
            prompt = create_numerical_prompt(needles_used=shuffled_needles, hay=hay)
            correct_model_ans = generate_correct_model_numerical_answer(needles, shuffled_needles, needle_cores, ans)
        elif q_type == 'code':
            prompt = create_code_prompt(needles_used=shuffled_needles, hay=hay)
            correct_model_ans = generate_correct_model_code_answer(needles, shuffled_needles, needle_cores, ans)

        result_verbose = {'prompt': prompt,
                          'correct-model-ans': correct_model_ans}
        ft_data.append(result_verbose)

    # Save eval results
    write_jsonl(out_fp, list_generator(ft_data))
    print('Created finetuning data, saved to', out_fp)

    return ft_data


# formats finetuning data into jsonl file for together finetuning
def create_together_ft_data(ft_data, save_fp):
    data_list = []
    for item in ft_data:
        data = f"[INST]{item['prompt']}[/INST]{item['correct-model-ans']}"
        data_list.append(data)

    write_jsonl(save_fp, list_generator(data_list))

    return data_list


def main():
    # data = create_progressive_needles_finetuning_data(num_needles_max=6,
    #                                             needle_func=create_code_needles,
    #                                             num_qs=1000,
    #                                             q_type='code',
    #                                             out_fp='test/testStuff5.jsonl',
    #                                             num_tokens_hay_max=20000,
    #                                             seed=42**2)

    # ppr.pprint(data)

    # create_together_ft_data(ft_data=data, save_fp='finetuning_data/code_needleMax6_hayMax20k_1000qs.jsonl')
    #
    # data = create_progressive_needles_finetuning_data(num_needles_max=6,
    #                                                   needle_func=create_numerical_needles,
    #                                                   num_qs=1000,
    #                                                   q_type='numerical',
    #                                                   out_fp='test/testStuff6.jsonl',
    #                                                   num_tokens_hay_max=20000,
    #                                                   seed=42**2)

    # ppr.pprint(data)

    # create_together_ft_data(ft_data=data, save_fp='finetuning_data/numerical_needleMax6_hayMax20k_1000qs.jsonl')

    # results = create_progressive_needles_eval_data(num_needles=2,
    #                                                needle_func=create_numerical_needles,
    #                                                num_qs=100,
    #                                                q_type='numerical',
    #                                                out_fp='evaluation_data/numerical_2needles_100qs.jsonl',
    #                                                hay_max_exclusive=21,
    #                                                seed=42)
    # results = create_progressive_needles_eval_data(num_needles=2,
    #                                                needle_func=create_code_needles,
    #                                                num_qs=100,
    #                                                q_type='code',
    #                                                out_fp='evaluation_data/code_2needles_100qs.jsonl',
    #                                                hay_max_exclusive=21,
    #                                                seed=42)
    # results = create_progressive_needles_eval_data(num_needles=4,
    #                                                needle_func=create_numerical_needles,
    #                                                num_qs=100,
    #                                                q_type='numerical',
    #                                                out_fp='evaluation_data/numerical_4needles_100qs.jsonl',
    #                                                hay_max_exclusive=21,
    #                                                seed=42)
    # results = create_progressive_needles_eval_data(num_needles=4,
    #                                                needle_func=create_code_needles,
    #                                                num_qs=100,
    #                                                q_type='code',
    #                                                out_fp='evaluation_data/code_4needles_100qs.jsonl',
    #                                                hay_max_exclusive=21,
    #                                                seed=42)

    results = create_progressive_needles_eval_data(num_needles=1,
                                                   needle_func=create_numerical_needles,
                                                   num_qs=100,
                                                   q_type='numerical',
                                                   out_fp='evaluation_data/numerical_1needles_100qs.jsonl',
                                                   hay_max_exclusive=21,
                                                   seed=42)

    results = create_progressive_needles_eval_data(num_needles=1,
                                                   needle_func=create_code_needles,
                                                   num_qs=100,
                                                   q_type='code',
                                                   out_fp='evaluation_data/code_1needles_100qs.jsonl',
                                                   hay_max_exclusive=21,
                                                   seed=42)

    # ppr.pprint(results)


if __name__ == '__main__':
    main()
