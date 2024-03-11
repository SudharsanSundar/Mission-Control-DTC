from model_class import GPT, TogModel, Claude
from io_processing import stream_jsonl, write_jsonl, list_generator
import random
import pprint as ppr
from pprint import PrettyPrinter
import tiktoken
import os

pp = ppr.PrettyPrinter(indent=4)
# All modern models
gpt_encoding = tiktoken.get_encoding('cl100k_base')


# Get specified number of tokens from given haystack file (whole words)
def get_hay(file: str, tokens: int):
    percent = tokens / 494000

    words = file.split(' ')
    new_words = words[:int(percent * len(words))]

    hay = ' '.join(new_words)

    return hay, len(gpt_encoding.encode(text=hay))


def get_code_hay(file: str, tokens: int):
    percent = tokens / 30290

    funcs = file.split('- / - / - /')
    new_funcs = funcs[:int(percent * len(funcs))]

    hay = '\n\n- / -\n\n'.join(new_funcs)

    return hay, len(gpt_encoding.encode(text=hay))


# Solver for the numerical needles task
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


# Create numerical needles
def numerical_needles(num_needles):
    ops = [' plus ', ' minus ']
    vals = []
    vals_num = []
    min = 0
    max = 10

    # Create needle operations. e.g. one op is "5, plus", the next is "1, minus" etc.
    for i in range(num_needles):
        num = random.randint(min, max)
        op = random.choice(ops)

        prefix = None
        if i < num_needles - 1:
            if 'plus' in op:
                prefix = str(num) + op
            elif 'minus' in op:
                prefix = op + str(num)
            vals_num.append([num, op])
        else:
            prefix = str(num)
            vals_num.append([num, None])

        vals.append(prefix)

    # Get correct answer (assuming each needle feeds into the previous, and we ask for value of needle 0)
    answer = rec_solve(vals_num)

    # Create needle sentences
    ordered_needles = []

    for i, val in zip(range(num_needles), vals):
        needle = None

        if i < num_needles - 1:
            if 'plus' in val:
                needle = "The value of Needle " + str(i) + " is " + val + "the value of Needle " + str(i+1) + '.'
            elif 'minus' in val:
                needle = "The value of Needle " + str(i) + " is " + "the value of Needle " + str(i+1) + val + '.'
        else:
            needle = "The value of Needle " + str(i) + " is " + val[0] + '.'

        ordered_needles.append(needle)

    # Shuffle needles as desired
    shuffled_needles = ordered_needles.copy()
    random.shuffle(shuffled_needles)

    return ordered_needles, shuffled_needles, answer


# Create code needles
def code_needles(num_needles):
    ops = [' plus ', ' minus ']
    vals = []
    vals_num = []
    min = 0
    max = 10

    # Create needle operations. e.g. one op is "5, plus", the next is "1, minus" etc.
    for i in range(num_needles):
        num = random.randint(min, max)
        op = random.choice(ops)

        prefix = None
        if i < num_needles - 1:
            if 'plus' in op:
                prefix = str(num) + op
            elif 'minus' in op:
                prefix = op + str(num)
            vals_num.append([num, op])
        else:
            prefix = str(num)
            vals_num.append([num, None])

        vals.append(prefix)

    # Get correct answer (assuming each needle feeds into the previous, and we ask for value of needle 0)
    answer = rec_solve(vals_num)

    # Create needle sentences
    ordered_needles = []

    for i, val_num in zip(range(num_needles), vals_num):
        needle = None

        if i < num_needles - 1:
            if 'plus' in val_num[1]:
                needle = "\n\ndef get_value_of_needle_" + str(i) + "():\n\treturn get_value_of_needle_" + str(i + 1) + "() + " + str(val_num[0]) + "\n\n"
            elif 'minus' in val_num[1]:
                needle = "\n\ndef get_value_of_needle_" + str(i) + "():\n\treturn get_value_of_needle_" + str(i + 1) + "() - " + str(val_num[0]) + "\n\n"
        else:
            needle = "\n\ndef get_value_of_needle_" + str(i) + "():\n\treturn " + str(val_num[0]) + "\n\n"

        ordered_needles.append(needle)

    # Shuffle needles as desired
    shuffled_needles = ordered_needles.copy()
    random.shuffle(shuffled_needles)

    return ordered_needles, shuffled_needles, answer


# Create prompt for needles only.
def create_only_numerical_needle_prompt(needles):
    # Prompt form: [use info to answer q][q][info][again use info to answer q][think step by step]
    preamble = "Please use the following information to correctly answer the following query.\n\n"
    query = 'What is the value of Needle 0?\n\n'
    engr = "Let's think step by step."
    needle_str = '\n'.join(needles) + '\n\n'

    prompt = preamble + "Query: " + query + "Information:\n" + needle_str + "Again, please use the above information to correctly answer the following query: " + query + engr

    return prompt


# Create prompt for needles hidden in haystack.
def create_hay_numerical_needle_prompt(hay, needles):
    # Prompt form: [use info to answer q][q][info][again use info to answer q][think step by step]
    hay_lines = hay.split(' ')
    num_needles = len(needles)
    rand_pos = random.sample(range(0, len(hay_lines) - 1), num_needles)
    rand_pos.sort()

    for needle, pos in zip(needles, rand_pos):
        needle_fmt = '(' + needle + ')'
        hay_lines.insert(pos, needle_fmt)

    hay_w_needles = ' '.join(hay_lines)

    preamble = "Please use the following information to correctly answer the following query.\n\n"
    query = 'What is the value of Needle 0?\n\n'
    engr = "Let's think step by step."

    prompt = preamble + "Query: " + query + "Information:\n" + hay_w_needles + "\n\nAgain, please use the above information to correctly answer the following query: " + query + engr

    return prompt


# Create prompt for needles only.
def create_only_code_needle_prompt(needles):
    # Prompt form: [use info to answer q][q][info][again use info to answer q][think step by step]
    preamble = "Please use the following code to correctly answer the following query.\n\n"
    query = 'What is the value returned when calling get_value_of_needle_0()?\n\n'
    engr = "Let's think step by step."
    needle_str = ''.join(needles) + '\n\n'

    prompt = preamble + "Query: " + query + "Code:" + needle_str[1:] + "Again, please use the above code to correctly answer the following query: " + query + engr

    return prompt


# Create prompt for needles hidden in haystack.
def create_hay_code_needle_prompt(hay, needles):
    # Prompt form: [use info to answer q][q][info][again use info to answer q][think step by step]
    hay_lines = hay.split('\n\n- / -\n\n')
    num_needles = len(needles)
    rand_pos = random.sample(range(0, len(hay_lines) - 1), num_needles)
    rand_pos.sort()

    for needle, pos in zip(needles, rand_pos):
        needle_fmt = needle
        hay_lines.insert(pos, needle_fmt)

    hay_w_needles = ''.join(hay_lines)

    preamble = "Please use the following code to correctly answer the following query.\n\n"
    query = 'What is the value returned when calling get_value_of_needle_0()?\n\n'
    engr = "Let's think step by step."

    prompt = preamble + "Query: " + query + "Code:\n" + hay_w_needles + "\n\nAgain, please use the above code to correctly answer the following query: " + query + engr

    return prompt


# Get numerical answer of the model. Gets the final number in last word/contiguous character sequence.
# Works on "10." "9." "**5.**" etc.
def extract_num(ans):
    words = ans.split(' ')
    final_word = words[-1]
    cleaned_num = ''

    for char in final_word:
        if char.isdigit() or char == '-':
            cleaned_num += char
    try:
        # num_ans = int(final_word[:-1])
        return int(cleaned_num)
    except ValueError:
        return 'Last word not numerical value: ' + final_word


# Runs eval on progressive needles task for a given model and problem formulation.
def run_needle_eval(num_needles: int,
                    needle_func,
                    num_tokens_hay: int,
                    corpus: str,
                    num_qs: int,
                    q_type: str,
                    out_fp: str,
                    model):
    random.seed(42)

    if q_type == 'numerical':
        hay, hay_tokens = get_hay(corpus, num_tokens_hay)
    elif q_type == 'code':
        hay, hay_tokens = get_code_hay(corpus, num_tokens_hay)
    print('ACTUAL HAY TOKENS: ', hay_tokens)

    eval_results = []
    num_right_raw = 0
    num_right_hay = 0

    for i in range(num_qs):
        # Generate needles and corresponding prompts
        needles, shuffled_needles, ans = needle_func(num_needles)
        if q_type == 'numerical':
            needle_only_prompt = create_only_numerical_needle_prompt(shuffled_needles)
            haystack_prompt = create_hay_numerical_needle_prompt(hay, shuffled_needles)
        elif q_type == 'code':
            needle_only_prompt = create_only_code_needle_prompt(shuffled_needles)
            haystack_prompt = create_hay_code_needle_prompt(hay, shuffled_needles)

        # Get model answers
        raw_ans = model.answer_txt(needle_only_prompt)  # Temp 0
        hay_ans = model.answer_txt(haystack_prompt)
        raw_ans_num = extract_num(raw_ans)
        hay_ans_num = extract_num(hay_ans)

        # Score answers
        if type(raw_ans_num) is int:
            num_right_raw += 1 if raw_ans_num == ans else 0
        if type(hay_ans_num) is int:
            num_right_hay += 1 if hay_ans_num == ans else 0

        # Save details of the example
        result_verbose = {'needles-only-prompt': needle_only_prompt,
                          'haystack-prompt': haystack_prompt,
                          'needles-only-ans': raw_ans,
                          'needles-only-ans-num': raw_ans_num,
                          'haystack-ans': hay_ans,
                          'haystack-ans-num': hay_ans_num,
                          'correct-ans': ans}
        eval_results.append(result_verbose)

        # Print update
        result = {'needles-only-ans': raw_ans,
                  'haystack-ans': hay_ans}
        pp.pprint(result)
        print('correct:', ans, '| needles only:', raw_ans_num, '(total', num_right_raw,  ') | haystack: ', hay_ans_num, '(total', num_right_hay, ')')

    # Save eval results
    write_jsonl(out_fp, list_generator(eval_results))

    return num_right_raw / num_qs, num_right_hay / num_qs, eval_results


def main():
    with open('../progressive_needles/hay_data/dost.txt', 'r') as f:
        novel = f.read()

    # with open('progressive_needles/hay_data/tb.txt', 'r') as f:
    #     tb = f.read()
    #
    with open('../progressive_needles/hay_data/human_eval_concat.txt', 'r') as f: # with open('progressive_needles/hay_data/human_eval_concat.txt', 'r') as f:
        code = f.read()

    gpt3p5 = GPT(model='gpt-3.5-turbo-0125')
    gpt4 = GPT(model='gpt-4-0125-preview')
    mixtral54BIns = TogModel(model='mistralai/Mixtral-8x7B-Instruct-v0.1')
    claude3Sonnet = Claude()
    claude3Opus = Claude(model="claude-3-opus-20240229")

    num_needles = 15
    num_tokens_hay = 10000
    num_qs = 50

    # acc_raw, acc_hay, results = run_needle_eval(num_needles=num_needles,
    #                                             needle_func=numerical_needles,
    #                                             num_tokens_hay=num_tokens_hay,
    #                                             corpus=novel,
    #                                             num_qs=num_qs,
    #                                             q_type='numerical',
    #                                             out_fp='progressive_needles/results/numericalProgNeedles_gpt3p5_15needles_10kT_50qs_seed42.jsonl',
    #                                             model=gpt3p5)
    # pp.pprint(results)
    # print(acc_raw, acc_hay)

    # num_needles = 10
    # num_tokens_hay = 10000
    # num_qs = 50
    #
    # acc_raw, acc_hay, results = run_needle_eval(num_needles=num_needles,
    #                                             needle_func=code_needles,
    #                                             num_tokens_hay=num_tokens_hay,
    #                                             corpus=code,
    #                                             num_qs=num_qs,
    #                                             q_type='code',
    #                                             out_fp='../progressive_needles/results/codeProgNeedles_gpt3p5_10needles_10kT_50qs_seed42.jsonl',
    #                                             model=gpt3p5)
    # pp.pprint(results)
    # print(acc_raw, acc_hay)

    # _, shuff, ans = code_needles(5)
    # code_hay, hay_tok = get_code_hay(code, 5000)
    # print('REAL TOK: ', hay_tok)
    #
    # print(create_hay_code_needle_prompt(code_hay, shuff))
    # print(create_only_code_needle_prompt(shuff), ans)

    # _, shuff, ans = numerical_needles(5)
    # hay, hay_tok = get_hay(code, 5000)
    # print('REAL TOK: ', hay_tok)
    #
    # print(create_hay_numerical_needle_prompt(hay, shuff))
    # print(create_only_numerical_needle_prompt(shuff), ans)

    hay, hay_tok = get_hay(novel, 5000)
    print(hay_tok)
    hay, hay_tok = get_hay(novel, 10000)
    print(hay_tok)





if __name__ == "__main__":
    main()