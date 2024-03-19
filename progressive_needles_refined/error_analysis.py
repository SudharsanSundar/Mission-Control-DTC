from model_class import GPT, TogModel, Claude
from io_processing import stream_jsonl, write_jsonl, list_generator, jsonl_to_list
import random
import pprint as ppr
from pprint import PrettyPrinter
import tiktoken
import os
import numpy as np
from matplotlib import pyplot as plt

pp = ppr.PrettyPrinter(indent=4)
# All modern models
gpt_encoding = tiktoken.get_encoding('cl100k_base')


def check_error_subsets(easy_key, hard_key, data_fp):
    data = jsonl_to_list(data_fp)

    results = {'easy_beats_hard': 0,
               'hard_beats_easy': 0,
               'both_same': 0}

    for datum in data:
        correct_ans = datum['correct-ans']
        easy_ans = datum[easy_key]
        hard_ans = datum[hard_key]

        if easy_ans == hard_ans or (easy_ans != correct_ans and hard_ans != correct_ans):
            results['both_same'] += 1
        elif easy_ans == correct_ans:
            results['easy_beats_hard'] += 1
        elif hard_ans == correct_ans:
            results['hard_beats_easy'] += 1

    ppr.pprint(results)

    return results


def check_absolute_deviation(easy_key, hard_key, data_fp):
    data = jsonl_to_list(data_fp)

    results = {'easy': 0,
               'hard': 0,
               }

    count_easy = 0
    count_hard = 0

    for datum in data:
        correct_ans = datum['correct-ans']
        easy_ans = datum[easy_key]
        hard_ans = datum[hard_key]

        try:
            results['easy'] += abs(int(correct_ans) - int(easy_ans))
            count_easy += 1
        except Exception as e:
            print('couldnt grade:', easy_ans)
            do_nothing = 0


        try:
            results['hard'] += abs(int(correct_ans) - int(hard_ans))
            count_hard += 1
        except Exception as e:
            # print('couldnt grade:', hard_ans)
            do_nothing = 0

    # ppr.pprint(results)
    results['easy'] /= count_easy
    results['hard'] /= count_hard
    ppr.pprint(results)
    # print('---')

    return results


def check_if_all_needles_retrieved(easy_key, hard_key, data_fp):
    data = jsonl_to_list(data_fp)

    results = {'easy': 0,
               'hard': 0,}

    if 'numerical' in data_fp:
        for datum in data:
            correct_ans = datum['correct-ans']
            easy_ans = datum[easy_key[:-4]]
            hard_ans = datum[hard_key[:-4]]

            if 'needle 0' in easy_ans.lower() and 'needle 1' in easy_ans.lower():
                if '4' in data_fp:
                    results['easy'] += 1 if 'needle 2' in easy_ans.lower() and 'needle 3' in easy_ans.lower() else 0
                else:
                    results['easy'] += 1

            if 'needle 0' in hard_ans.lower() and 'needle 1' in hard_ans.lower():
                if '4' in data_fp:
                    results['hard'] += 1 if 'needle 2' in hard_ans.lower() and 'needle 3' in hard_ans.lower() else 0
                else:
                    results['hard'] += 1
    elif 'code' in data_fp:
        for datum in data:
            correct_ans = datum['correct-ans']
            easy_ans = datum[easy_key[:-4]]
            hard_ans = datum[hard_key[:-4]]

            if '_0' in easy_ans.lower() and '_1' in easy_ans.lower():
                if '4' in data_fp:
                    results['easy'] += 1 if '_2' in easy_ans.lower() and '_3' in easy_ans.lower() else 0
                elif '2' in data_fp:
                    results['easy'] += 1
            else:
                print(easy_ans)

            if '_0' in hard_ans.lower() and '_1' in hard_ans.lower():
                if '4' in data_fp:
                    results['hard'] += 1 if '_2' in hard_ans.lower() and '_3' in hard_ans.lower() else 0
                elif '2' in data_fp:
                    results['hard'] += 1
            else:
                print(hard_ans)

    ppr.pprint(results)

    return results


def check_refusals(easy_key, hard_key, data_fp):
    data = jsonl_to_list(data_fp)

    results = {'easy': 0,
               'hard': 0,}

    for datum in data:
        correct_ans = datum['correct-ans']
        easy_ans_num = datum[easy_key]
        hard_ans_num = datum[hard_key]
        easy_ans = datum[easy_key[:-4]]
        hard_ans = datum[hard_key[:-4]]

        if easy_ans_num != correct_ans:
            if 'cannot' in easy_ans.lower() or 'unable' in easy_ans.lower() or 'not provided' in easy_ans.lower():
                results['easy'] += 1
                # print(easy_ans)
        if hard_ans_num != correct_ans:
            if 'cannot' in hard_ans.lower() or 'unable' in hard_ans.lower() or 'not provided' in hard_ans.lower():
                results['hard'] += 1
                # print(hard_ans)

    ppr.pprint(results)


def main():
    # Are the ones right in hard settings a subset of the ones right in other easier settings?
    needles_only = 'needles-only-prompt-model-ans-num'
    hay2k = 'hay2k-prompt-model-ans-num'
    hay7k = 'hay7k-prompt-model-ans-num'
    hay12k = 'hay12k-prompt-model-ans-num'
    hay20k = 'hay20k-prompt-model-ans-num'

    mixtral_numerical_2 = '/Users/sudharsansundar/Mission-Control-DTC/progressive_needles_refined/evaluation_results/mixtral8x7B_numerical_2needles_results.jsonl'
    mixtral_numerical_4 = '/Users/sudharsansundar/Mission-Control-DTC/progressive_needles_refined/evaluation_results/mixtral8x7B_numerical_4needles_results.jsonl'
    mixtral_code_2 = '/Users/sudharsansundar/Mission-Control-DTC/progressive_needles_refined/evaluation_results/mixtral8x7B_code_2needles_results.jsonl'
    mixtral_code_4 = '/Users/sudharsansundar/Mission-Control-DTC/progressive_needles_refined/evaluation_results/mixtral8x7B_code_4needles_results.jsonl'
    gpt3p5_numerical_2 = ''
    gpt3p5_numerical_4 = ''
    gpt3p5_code_2 = ''
    gpt3p5_code_4 = ''
    gpt4_numerical_4 = '/Users/sudharsansundar/Mission-Control-DTC/progressive_needles_refined/evaluation_results/gpt4_numerical_4needles_results.jsonl'
    gpt4_code_4 = ''

    print('MIXTRAL')
    # check if direct degradation, i.e. haystack performance is question subset of needles only performance
    # check_error_subsets(needles_only, hay2k, mixtral_numerical_2)
    # check_error_subsets(needles_only, hay7k, mixtral_numerical_2)
    # check_error_subsets(needles_only, hay12k, mixtral_numerical_2)
    check_error_subsets(needles_only, hay2k, mixtral_numerical_4)
    check_error_subsets(needles_only, hay7k, mixtral_numerical_4)
    check_error_subsets(needles_only, hay12k, mixtral_numerical_4)
    # check_error_subsets(needles_only, hay2k, mixtral_code_2)
    # check_error_subsets(needles_only, hay7k, mixtral_code_2)
    # check_error_subsets(needles_only, hay12k, mixtral_code_2)
    # check_error_subsets(needles_only, hay2k, mixtral_code_4)
    # check_error_subsets(needles_only, hay7k, mixtral_code_4)
    # check_error_subsets(needles_only, hay12k, mixtral_code_4)

    print('MIXTRAL')

    # check if it recognizes all needles are needed
    # check_if_all_needles_retrieved(needles_only, hay2k, mixtral_numerical_2)
    # check_if_all_needles_retrieved(needles_only, hay7k, mixtral_numerical_2)
    # check_if_all_needles_retrieved(needles_only, hay12k, mixtral_numerical_2)
    # check_if_all_needles_retrieved(needles_only, hay2k, mixtral_numerical_4)
    # check_if_all_needles_retrieved(needles_only, hay7k, mixtral_numerical_4)
    # check_if_all_needles_retrieved(needles_only, hay12k, mixtral_numerical_4)
    # check_if_all_needles_retrieved(needles_only, hay2k, mixtral_code_2)
    # check_if_all_needles_retrieved(needles_only, hay7k, mixtral_code_2)
    # check_if_all_needles_retrieved(needles_only, hay12k, mixtral_code_2)
    # check_if_all_needles_retrieved(needles_only, hay2k, mixtral_code_4)
    # check_if_all_needles_retrieved(needles_only, hay7k, mixtral_code_4)
    # check_if_all_needles_retrieved(needles_only, hay12k, mixtral_code_4)

    print('MIXTRAL')

    # check if continuous metric supports assertion
    # check_absolute_deviation(needles_only, hay2k, mixtral_numerical_2)
    # check_absolute_deviation(needles_only, hay7k, mixtral_numerical_2)
    # check_absolute_deviation(needles_only, hay12k, mixtral_numerical_2)
    check_absolute_deviation(needles_only, hay2k, mixtral_numerical_4)
    check_absolute_deviation(needles_only, hay7k, mixtral_numerical_4)
    check_absolute_deviation(needles_only, hay12k, mixtral_numerical_4)
    # check_absolute_deviation(needles_only, hay2k, mixtral_code_2)
    # check_absolute_deviation(needles_only, hay7k, mixtral_code_2)
    # check_absolute_deviation(needles_only, hay12k, mixtral_code_2)
    # check_absolute_deviation(needles_only, hay2k, mixtral_code_4)
    # check_absolute_deviation(needles_only, hay7k, mixtral_code_4)
    # check_absolute_deviation(needles_only, hay12k, mixtral_code_4)

    print('MIXTRAL')

    # check to see how many wrong answers are refusals
    # check_refusals(needles_only, hay2k, mixtral_numerical_2)
    # check_refusals(needles_only, hay7k, mixtral_numerical_2)
    # check_refusals(needles_only, hay12k, mixtral_numerical_2)
    check_refusals(needles_only, hay2k, mixtral_numerical_4)
    check_refusals(needles_only, hay7k, mixtral_numerical_4)
    check_refusals(needles_only, hay12k, mixtral_numerical_4)
    # check_refusals(needles_only, hay2k, mixtral_code_2)
    # check_refusals(needles_only, hay7k, mixtral_code_2)
    # check_refusals(needles_only, hay12k, mixtral_code_2)
    # check_refusals(needles_only, hay2k, mixtral_code_4)
    # check_refusals(needles_only, hay7k, mixtral_code_4)
    # check_refusals(needles_only, hay12k, mixtral_code_4)

    # # # GPT 4
    print('GPT-4')
    check_error_subsets(needles_only, hay20k, gpt4_numerical_4)
    check_if_all_needles_retrieved(needles_only, hay20k, gpt4_numerical_4)
    check_absolute_deviation(needles_only, hay20k, gpt4_numerical_4)
    check_refusals(needles_only, hay20k, gpt4_numerical_4)


if __name__ == '__main__':
    main()
