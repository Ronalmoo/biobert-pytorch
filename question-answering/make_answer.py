import json


if __name__ == "__main__":

    with open('BioASQ-test-factoid-6b-5.json') as f:
        file = json.load(f)
    # print(file['data'][0]['paragraphs'][0]['context'])

    for data in file['data']:
        print(data['paragraphs'])
        for context in data['paragraphs']:
            for qa in context['qas']:
                qa['answers'] = []
                print(qa)
                with open("BioASQ-test-factoid-6b-5_answers.json", "w") as json_file:
                    json.dump(file, json_file, indent=2)

