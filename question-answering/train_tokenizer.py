from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
from tokenizers.models import BPE
import json

def build_new_vocab():

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.pre_tokenizer = Whitespace()


    # files = [f"/daintlab/home/moo/NLU/biobert-pytorch/datasets/QA/BioASQ/BioASQ-{split}-factoid-7b.json" for split in ["train_split", "dev"]]
    files = "/daintlab/home/moo/NLU/biobert-pytorch/datasets/QA/BioASQ/BioASQ-train-factoid-7b.json"


    with open(files) as f:
        file = json.load(f)
    contexts = []
    for question in file['data']:
        for paragraph in question['paragraphs']:
            contexts.append(paragraph['context'])

    tokenizer.train_from_iterator(contexts, trainer)
    additional_vocab = [k for k, v in tokenizer.get_vocab().items()]

    tokenizer.save("tokenizer/tokenizer-bioasq.json")
    return additional_vocab