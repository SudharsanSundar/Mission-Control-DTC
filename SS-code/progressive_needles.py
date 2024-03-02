from model_class import GPT, MetaRNN, Corpus, TogModel
import random
import pprint as ppr
from pprint import PrettyPrinter

pp = ppr.PrettyPrinter(indent=4)


def process_dost(txt_path):
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


# get sized chunk
def get_hay(file, tokens):
    percent = tokens / 494000

    words = file.split(' ')
    new_words = words[:int(percent * len(words))]

    hay = ' '.join(new_words)

    return hay


def rec_solve(vals):
    # print(vals)
    if len(vals) == 1:
        # print(vals[0][0])
        return vals[0][0]
    else:
        cur = vals[0]
        ans = rec_solve(vals[1:])
        if cur[1] == ' plus ':
            # print(cur[0], '+', ans)
            return ans + cur[0]
        elif cur[1] == ' minus ':
            # print('-', cur[0], '+', ans)
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

    # Create needle operations
    for i in range(num_needles):
        num = random.randint(min, max)
        op = random.choice(ops)

        prefix = None
        if i < num_needles - 1:
            if ' plus ' in op:
                prefix = str(num) + op
            elif 'minus' in op:
                prefix = op + str(num)
            vals_num.append([num, op])
        else:
            prefix = str(num)
            vals_num.append([num, None])

        vals.append(prefix)

    # Get correct answer
    answer = rec_solve(vals_num)

    # Create needles
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


# create needles
def create_only_needle_prompt(needles):
    preamble = "Please use the following information to correctly answer the following query.\n\n"
    query = 'What is the value of Needle 0?\n\n'
    engr = "Let's think step by step."
    # needles.reverse()
    needle_str = '\n'.join(needles) + '\n\n'

    prompt = preamble + "Query: " + query + "Information:\n" + needle_str + "Again, please use the above information to correctly answer the following query: " + query + engr

    return prompt


def create_hay_needle_prompt(hay, needles):
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
        return 'Last word not numerical value'


def run_numerical_needle_eval(num_needles, num_tokens, corpus, num_qs, model):
    random.seed(42)

    hay = get_hay(corpus, num_tokens)
    eval_results = []
    num_right_raw = 0
    num_right_hay = 0

    for i in range(num_qs):
        needles, shuff_needles, ans = numerical_needles(num_needles)
        needle_only_prompt = create_only_needle_prompt(shuff_needles)
        haystack_prompt = create_hay_needle_prompt(hay, shuff_needles)

        raw_ans = model.answer_txt(needle_only_prompt)  # TODO: make generation up to 1k tokens long, TEMP 0, etc.
        hay_ans = model.answer_txt(haystack_prompt)

        raw_ans_num = extract_num(raw_ans)
        hay_ans_num = extract_num(hay_ans)
        if type(raw_ans_num) is int:
            num_right_raw += 1 if raw_ans_num == ans else 0
        if type(hay_ans_num) is int:
            num_right_hay += 1 if hay_ans_num == ans else 0

        eval_results.append({'needles-only-prompt': needle_only_prompt,
                             'haystack-prompt': haystack_prompt,
                             'needles-only-ans': raw_ans,
                             'needles-only-ans-num': raw_ans_num,
                             'haystack-ans': hay_ans,
                             'haystack-ans-num': hay_ans_num,
                             'correct-ans': ans})

        result = {'needles-only-ans': raw_ans,
                  'haystack-ans': hay_ans,
                 }
        pp.pprint(result)
        print('correct:', ans, '| needles only:', raw_ans_num, '(total', num_right_raw,  ') | haystack: ', hay_ans_num, '(total', num_right_hay, ')')

    return num_right_raw / num_qs, num_right_hay / num_qs, eval_results


def main():
    with open('dost.txt', 'r') as f:
        dost = f.read()

    with open('tb.txt', 'r') as f:
        tb = f.read()

    gpt3p5 = GPT(model='gpt-3.5-turbo-0125')
    gpt4 = GPT(model='gpt-4-0125-preview')
    mixtral54BIns = TogModel(model='mistralai/Mixtral-8x7B-Instruct-v0.1')

    tokens16k = 16000
    hay = get_hay(dost, 1000)
    # print(hay)

    # num_needles = 10
    # needles, shuffled_needles, ans = numerical_needles(num_needles)
    # for needle in needles:
    #     print(needle)
    # print(ans)

    # print(create_only_needle_prompt(needles))
    # print(create_only_needle_prompt(shuffled_needles))
    # print(ans)

    # print(create_hay_needle_prompt(hay, needles))
    # print(ans)

    num_needles = 5
    num_tokens = 5000
    num_qs = 50

    acc1, acc2, results = run_numerical_needle_eval(num_needles, num_tokens, tb, num_qs, gpt3p5)
    # pp.pprint(results)
    # print(acc1, acc2)


if __name__ == "__main__":
    main()
