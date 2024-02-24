# from model_class import GPT, MetaRNN, Corpus
# from embedding_utils import FaissIndex, OpenAIEmbeddingModel, cosine_similarity
# import pandas as pd
# import json
#
# TODO: Salman, you'll need to take care of this. I think I'd waste too much time getting this all set up.
# def load_quality_data(data_path):
#     df = pd.DataFrame()
#     with open(data_path, 'r') as f:
#         for line in f:
#             example = json.loads(line)
#
#             exampleDF = pd.DataFrame(example)
#
#             for question in example['questions']:
#
#                 new_row = pd.concat([exampleDF.loc[:, exampleDF.columns != 'questions'], pd.DataFrame(question)])
#
#             # df.append(example, ignore_index=True)
#             df = pd.concat([df, pd.DataFrame(example)], ignore_index=True)
#             # print(df.tail())
#
#         print(df.columns.tolist())
#
#     print(df.tail(n=20))
#     print(df['questions'][0])
#
#
#
# load_quality_data('../data/quality.train')
