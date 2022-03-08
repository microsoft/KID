import os
import pickle
from typing import Any, List

import marisa_trie
import neuralcoref
import pandas as pd
import spacy
import torch
from datasets import load_dataset
from openie import StanfordOpenIE
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.pipeline import EntityRuler
from spacy.tokens import Token
from tqdm.notebook import tqdm
from transformers import RagRetriever, RagTokenForGeneration, RagTokenizer

import KID

stop_words_getter = lambda token: token.is_stop or token.lower_ in STOP_WORDS \
                                                or token.lemma_ in STOP_WORDS

Token.set_extension('is_stop', getter=stop_words_getter, force=True)

nlp = spacy.load("en_core_web_sm")
ruler = EntityRuler(nlp)
nlp.add_pipe(ruler)

neuralcoref.add_to_pipe(nlp)

properties = {
    'openie.affinity_probability_cap': 1 / 3,
}

retriever_tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained(
    "facebook/rag-token-nq", index_name="legacy", use_dummy_dataset=False
)
rag_model = RagTokenForGeneration.from_pretrained(
    "facebook/rag-token-nq", retriever=retriever
)

kg_dataset = load_dataset("wiki_dpr", 'psgs_w100.nq.exact')


def return_retrieved_psgs(
    tokenizer: Any, retriever: Any, rag: Any, query: str, k_docs: int
):
    """Retrieve docs by query.

    :param tokenizer: the tokenizer of RAG
    :param rag: the RAG model loaded
    :param retriever: the retriever of RAG
    :param query: the question/context of the NLG task. A string.
    :param k_docs: the number of documents to be retrieved
    :return: two lists. [doc ids] and [doc scores].
    """
    inputs = tokenizer(query, return_tensors="pt")
    input_ids = inputs["input_ids"]

    question_hidden_states = rag.question_encoder(input_ids)[0]
    docs_dict = retriever(
        input_ids.numpy(),
        question_hidden_states.detach().numpy(),
        return_tensors="pt",
        n_docs=k_docs
    )
    doc_scores = torch.bmm(
        question_hidden_states.unsqueeze(1),
        docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
    ).squeeze(1)
    return docs_dict['doc_ids'].detach().cpu().numpy().tolist()[0], doc_scores.detach(
    ).cpu().numpy().tolist()[0]


def convert2kg(texts: List[str], client: Any):
    """Extract <subj., rel., obj.> triplets from list of texts.

    :param texts: a list of text
    :param client: the OpenIE client
    :return:
    """
    str_list = []
    for text in texts:
        resolved_text = nlp(text)._.coref_resolved
        for triple in client.annotate(resolved_text):
            str_list.append(triple['subject'] + ' ' + triple['object'])
    return str_list


def return_entities(sent: str):
    return [
        token.lemma_.lower() for token in nlp(sent)
        if token.pos_ in ['PROPN', 'NOUN'] and not token.is_stop
    ]


def create_external_graph(questions: List[str], n: int, task_path: str):
    """Creating knowledge trie by querying the knowledge source
    by a list of "questions".

    :param questions: the "context" of each NLG task. e.g., questions in QA.
    :param n: number of documents to be retrieved.
    :param task_path: the output path for the trie.
    :return: None. The knowledge trie will be stored as a pickle file.
    """
    full_str_list = []
    with StanfordOpenIE(properties=properties) as client:
        for question in tqdm(questions, total=len(questions)):
            kg_docs_ids, doc_scores = return_retrieved_psgs(
                retriever_tokenizer, retriever, rag_model, question, n
            )
            kg_docs_text = kg_dataset['train'][kg_docs_ids]['text']
            full_str_list.extend(convert2kg(kg_docs_text, client))
    trie = marisa_trie.Trie(full_str_list)

    with open(task_path + '/trie_' + str(n) + '.pickle', 'wb') as handle:
        pickle.dump(trie, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # change the task name and the path to the correct one
    # here is an example for ELI5 with num_doc = 5
    task_name = 'eli5'
    assets_path = os.path.join(list(KID.__path__)[0], "assets")
    task_path = os.path.join(assets_path, task_name)
    train_df = pd.read_csv(task_path + '/train.csv', header=0)
    create_external_graph(train_df['question'].tolist()[:], n=5, task_path=task_path)
