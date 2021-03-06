import json
import numpy as np
import pandas as pd
import string
import re

from pathlib import Path
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    get_cosine_schedule_with_warmup,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
    )
from transformers.data.processors.squad import SquadV1Processor


def get_predictions(qid):
    # given a question id (qas_id or qid), load the example, get the model outputs and generate an answer
    question = examples[qid_to_example_index[qid]].question_text
    context = examples[qid_to_example_index[qid]].context_text

    inputs = tokenizer.encode_plus(question, context, return_tensors='pt')

    outputs = model(**inputs)
    answer_start = torch.argmax(outputs[0])  # get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(outputs[1]) + 1 

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

    return answer


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))



def get_gold_answers(example):
    """helper function that retrieves all possible true answers from a squad2.0 example"""
    
    gold_answers = [answer["text"] for answer in example.answers if answer["text"]]

    # if gold_answers doesn't exist it's because this is a negative example - 
    # the only correct answer is an empty string
    if not gold_answers:
        gold_answers = [""]
        
    return gold_answers


def extract_q_and_a(factoid_path: Path):
    with factoid_path.open() as json_file:
        data = json.load(json_file)
    
    questions = data['data'][0]['paragraphs']
    data_rows = []

    for question in questions:
        context = question['context']
        for question_and_answer in question['qas']:
            question = question_and_answer['question']
            answers = question_and_answer['answers']

            for answer in answers:
                answer_text = answer['text']
                answer_start = answer['answer_start']
                answer_end = answer_start + len(answer_text)

                data_rows.append({
                    'question': question,
                    'context': context,
                    'answer_text': answer_text,
                    'answer_start': answer_start,
                    'answer_end': answer_end,
                })
    return pd.DataFrame(data_rows)


def aurc_eaurc(rank_conf, rank_corr):
    li_risk = []
    li_coverage = []
    risk = 0
    for i in range(len(rank_conf)):
        coverage = (i + 1) / len(rank_conf)
        li_coverage.append(coverage)

        if rank_corr[i] == 0:
            risk += 1

        li_risk.append(risk / (i + 1))

    r = li_risk[-1]
    risk_coverage_curve_area = 0
    optimal_risk_area = r + (1 - r) * np.log(1 - r)
    for risk_value in li_risk:
        risk_coverage_curve_area += risk_value * (1 / len(li_risk))

    aurc = risk_coverage_curve_area
    eaurc = risk_coverage_curve_area - optimal_risk_area

    print(f'* AURC\t\t{round(aurc * 1000, 2)}')
    print(f'* E-AURC\t{round(eaurc * 1000, 2)}')

    return aurc, eaurc

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
    # tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    # model = AutoModelForQuestionAnswering.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    pred_path = "/daintlab/home/moo/NLU/biobert-pytorch/question-answering/output/dmis-lab/biobert-base-cased-v1.1/bioasq_indomain_7b_F1/nbest_predictions_.json"
    golden_path = "/daintlab/home/moo/NLU/biobert-pytorch/datasets/QA/BioASQ/BioASQ-dev-factoid-7b.json"

    squad_pred = "/daintlab/home/moo/NLU/biobert-pytorch/question-answering/output/dmis-lab/biobert-base-cased-v1.1/bioasq_outdomain_7b_F1/nbest_predictions_.json"
    squad_golden = "/daintlab/data/NLU/squad/dev-v1.1.json"

    processor = SquadV1Processor()
    in_goldens = processor.get_dev_examples("/daintlab/home/moo/NLU/biobert-pytorch/datasets/QA/BioASQ", filename="BioASQ-dev-factoid-7b.json")
    out_goldens = processor.get_dev_examples("/daintlab/data/NLU/squad", filename="dev-v1.1.json")
    

    in_qid_to_example_index = {example.qas_id: i for i, example in enumerate(in_goldens)}
    in_qid_to_has_answer = {example.qas_id: bool(example.answers) for example in in_goldens}

    out_qid_to_example_index = {example.qas_id: i for i, example in enumerate(out_goldens)}
    out_qid_to_has_answer = {example.qas_id: bool(example.answers) for example in out_goldens}


    in_answer_qids = [qas_id for qas_id, has_answer in in_qid_to_has_answer.items() if has_answer]
    out_answer_qids = [qas_id for qas_id, has_answer in out_qid_to_has_answer.items() if has_answer]

   
    # tr_df = extract_q_and_a(Path(golden_path))
    # tr_df = extract_q_and_a(Path('/daintlab/data/NLU/squad/train-v1.1.json'))
     
    # dev_df = extract_q_and_a(Path(squad_golden))
    # dev_df = extract_q_and_a(Path('/daintlab/data/NLU/squad/dev-v1.1.json'))

    # tr_df = tr_df.drop_duplicates(subset=['context']).reset_index(drop=True)
    # dev_df = dev_df.drop_duplicates(subset=['context']).reset_index(drop=True)

    # print(tr_df)
    # print(dev_df)
    # print(tr_df['answer_text'].values[1])
    # normalized_answers = []
    # for i, _ in enumerate(tr_df['answer_text']):
    #     normalized_answers.append(normalize_answer(tr_df['answer_text'].values[i]))
    # print(normalized_answers)
    # print(len(examples))
    

    with open(pred_path) as in_pred:
        in_pred_file = json.load(in_pred)
        in_preds = []
        in_probs = []
        # print(pred_file)
        for value in in_pred_file.values():
            in_preds.append(value[0]['text'])
            in_probs.append(value[0]['probability'])

    
    with open(squad_pred) as out_pred:
        out_pred_file = json.load(out_pred)
        out_preds = []
        out_probs = []
        # print(pred_file)
        for value in out_pred_file.values():
            # if len(preds) == len(in_pred_file):
            #     break
            out_preds.append(value[0]['text'])
            out_probs.append(value[0]['probability'])
            
    in_df = pd.DataFrame(in_preds)
    in_df['prob'] = pd.DataFrame(in_probs)
    # df['out_prob'] = pd.DataFrame(out_probs)
    # in_goldens.extend(out_goldens)
    out_df = pd.DataFrame(out_preds)
    out_df['prob'] = pd.DataFrame(out_probs)


    in_answers, out_answers = [], []
    for i in range(len(in_goldens)):
        # print(qid_to_example_index[answer_qids[i]])
        in_answers.append(get_gold_answers(in_goldens[in_qid_to_example_index[in_answer_qids[i]]]))
    
    for i in range(len(out_goldens)):
        # print(qid_to_example_index[answer_qids[i]])
        out_answers.append(get_gold_answers(out_goldens[out_qid_to_example_index[out_answer_qids[i]]]))



    in_exact, out_exact = [], []
    for i, _ in enumerate(in_answers):
        if  in_preds[i] in in_answers[i]:
            in_exact.append(1)
        else:
            in_exact.append(0)
        # exact.append(candidated_answers[i] == preds[i])
    # print(exact.count(1) / len(exact))
    for i, _ in enumerate(out_answers):
        if  out_preds[i] in out_answers[i]:
            out_exact.append(1)
        else:
            out_exact.append(0)
    in_df['exact'] = pd.DataFrame(in_exact)
    out_df['exact'] = pd.DataFrame(out_exact)


    print(in_df)
    print(out_df)
    concat_df = pd.concat([in_df, out_df], ignore_index=True)
    concat_df.sort_values(by='prob', ascending=False, inplace=True)
    print(concat_df)
    aurc_eaurc(concat_df['prob'].values, concat_df['exact'].values)




    import sys; sys.exit(1)
    

    # print(goldens[0])
    # print(len(goldens))
                # for answer in qa['answers']:
                #     print(answer)
                    # goldens.append(answer['text'])
    


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
    print(preds)
    print(out_goldens)

    # import sys; sys.exit(1)
    
    with open(squad_pred) as out_pred:
        out_pred_file = json.load(out_pred)
        # out_preds = []
        # out_probs = []
        # print(pred_file)
        for value in out_pred_file.values():
            # if len(preds) == len(in_pred_file):
            #     break
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
    df.sort_values(by='prob', ascending=False, inplace=True)
    print(df)
    # print(out_goldens)



    aurc_eaurc(df['prob'].values, df['exact'].values)
    
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
    
