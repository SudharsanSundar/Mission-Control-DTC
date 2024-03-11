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

def check_subsets(file):
    shortRight_longWrong = 0
    shortRight_longRight = 0
    shortWrong_longWrong = 0
    shortWrong_longRight = 0
    raw_diff_short = 0
    raw_diff_long = 0

    for ex in file:
        shortAns = ex['needles-only-ans-num'] if type(ex['needles-only-ans-num']) == int else -10000
        longAns = ex['haystack-ans-num'] if type(ex['haystack-ans-num']) == int else -10000
        correctAns = ex['correct-ans']

        if shortAns == correctAns:
            if longAns == correctAns:
                shortRight_longRight += 1
            else:
                shortRight_longWrong += 1
                raw_diff_long += abs(correctAns - longAns) / abs(correctAns) if longAns != -10000 else 0
        else:
            if longAns == correctAns:
                shortWrong_longRight += 1
                raw_diff_short += abs(correctAns - shortAns) / abs(correctAns) if shortAns != -10000 else 0
            else:
                shortWrong_longWrong += 1
                raw_diff_short += abs(correctAns - shortAns) / abs(correctAns) if shortAns != -10000 else 0
                raw_diff_long += abs(correctAns - longAns) / abs(correctAns) if longAns != -10000 else 0

    print('short right, long wrong:', shortRight_longWrong)
    print('short wrong, long right:', shortWrong_longRight)
    print('both right:', shortRight_longRight)
    print('both wrong:', shortWrong_longWrong)
    print('raw abs dev, short:', raw_diff_short, raw_diff_short / 50)
    print('raw abs dev, long:', raw_diff_long, raw_diff_long / 50)
    print('total qs:', shortRight_longWrong + shortWrong_longRight + shortRight_longRight + shortWrong_longWrong)


def main():
    print(os.getcwd())
    claude3OpusAnsF = 'results/numericalProgNeedles_Claude3Opus_15needles_10kT_50qs_seed42.jsonl'
    gpt3p5AnsF = 'results/numericalProgNeedles_gpt3p5_15needles_10kT_50qs_seed42.jsonl'
    claude3OpusAns = stream_jsonl(claude3OpusAnsF)
    gpt3p5Ans = stream_jsonl(gpt3p5AnsF)

    check_subsets(claude3OpusAns)
    check_subsets(gpt3p5Ans)


if __name__ == "__main__":
    main()
