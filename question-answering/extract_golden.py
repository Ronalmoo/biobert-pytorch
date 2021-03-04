import json
import numpy as np
import pandas as pd

def extract_pred(pred_path):
    with open(pred_path) as pred:
        pred_file = json.load(pred)
        preds = []
        for i in pred_file.keys():
            preds.append(pred_file[i])
    return preds


def extract_golden(golden_path):
    with open(golden_path) as golden:
        golden_file = json.load(golden)
    # print(golden_file['data'][0]['paragraphs'][0]['qas'][0]['answers'][0]['text'])
    goldens = []

    for data in golden_file['data']:
        for paragraph in data['paragraphs']:
            for qa in paragraph['qas']:
                # print(qa['answers'][0]['text'])
                goldens.append(qa['answers'][0]['text'])
                # for answer in qa['answers']:
                #     goldens.append(answer['text'])
    return goldens


def sorted_results(goldens, preds):
    sorted_list = []
    for i, _ in enumerate(goldens[:len(preds)]):
        sorted_list.append(goldens[i] == preds[i])
    return sorted_list


# def squad_results(goldes, preds):
#     sorted_list = []
#     for i, _ in enumerate(goldens[:len(preds)]):
#         sorted_list.append(goldens[i] == preds[i])
#     return sorted_list



if __name__ == "__main__":
    # golden_path = "/daintlab/data/NLU/squad/dev-v1.1.json"
    # with open(golden_path) as golden:
    #     golden_file = json.load(golden)
    # # print(golden_file['data'][0]['paragraphs'][0]['qas'][0]['answers'][0]['text'])
    # goldens = []
    # # candidates = []
    # for data in golden_file['data']:
    #     for paragraph in data['paragraphs']:
    #         # print(paragraph)
    #         for qa in paragraph['qas']:
    #             for answer in qa['answers']:
    #                 print(answer.values())
                # print(qa['answers'])
                # goldens.append(qa['answers'])
    # ls = []
    # # print(goldens)
    # for i in goldens:
    #     # ls.append(i['text'])
    #     candidates = np.array(i).reshape(1, -1)
    #     print(candidates)
    #     if 'slug' in candidates:
    #         print('ss')
    # print(np.array(ls).reshape(1, -1))

    

    # print(goldens[0])
    # print(len(goldens))
                # for answer in qa['answers']:
                #     print(answer)
                    # goldens.append(answer['text'])
    



    # import sys; sys.exit(1)
    pred_path = "/daintlab/home/moo/NLU/biobert-pytorch/question-answering/output/bert-base-cased/bioasq_indomain_7b_F1/nbest_predictions_.json"
    golden_path = "/daintlab/home/moo/NLU/biobert-pytorch/datasets/QA/BioASQ/BioASQ-dev-factoid-7b.json"


    squad_pred = "/daintlab/home/moo/NLU/biobert-pytorch/question-answering/output/bert-base-cased/bioasq_outdomain_7b_F1/nbest_predictions_.json"
    squad_golden = "/daintlab/data/NLU/squad/dev-v1.1.json"

    # in_preds = extract_pred(pred_path)
    in_goldens = extract_golden(golden_path)
    # in_results = sorted_results(in_goldens, in_preds)

    # out_preds = extract_pred(squad_pred)
    out_goldens = extract_golden(squad_golden)
    # out_results = sorted_results(out_goldens, out_preds)

    # print(out_preds[:10])
    # print(out_goldens[:10])
    # print(out_results.count(True))

    with open(pred_path) as in_pred:
        in_pred_file = json.load(in_pred)
        preds = []
        probs = []
        # print(pred_file)
        for value in in_pred_file.values():
            preds.append(value[0]['text'])
            probs.append(value[0]['probability'])
    
    with open(squad_pred) as out_pred:
        out_pred_file = json.load(out_pred)
        # out_preds = []
        # out_probs = []
        # print(pred_file)
        for value in out_pred_file.values():
            preds.append(value[0]['text'])
            probs.append(value[0]['probability'])
    df = pd.DataFrame(preds)
    df['prob'] = pd.DataFrame(probs)
    # df['out_prob'] = pd.DataFrame(out_probs)
    in_goldens.extend(out_goldens)

    

    
    exact = []
    for i, _ in enumerate(in_goldens):
        exact.append(in_goldens[i] == preds[i])
    # print(exact)
    df['exact'] = pd.DataFrame(exact)
    print(df.sort_values(by='prob', ascending=False).head(100))
    
    # print(df.values)


            # preds.append(pred_file[i])
            # preds.append(pred_file[i][0]['text'])
    # print(preds)

    # import sys; sys.exit(1)
    
    # with open(golden_path) as golden:
    #     golden_file = json.load(golden)
    # # print(golden_file['data'][0]['paragraphs'][0]['qas'][0]['answers'][0]['text'])
    # goldens = []

    # for data in golden_file['data']:
    #     for paragraph in data['paragraphs']:
    #         for qa in paragraph['qas']:
    #             for answer in qa['answers']:
    #                 goldens.append(answer['text'])

    # print(goldens)
    # print(preds)
    
    # sorted_list = []
    # for i, _ in enumerate(goldens):
    #     sorted_list.append(goldens[i] == preds[i])
    # print(sorted_list[:200].count(True))
    
