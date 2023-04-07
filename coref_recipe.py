import time
from collections import defaultdict
from typing import Union, List, Dict

from numpy import ndarray
# from util import make_coref_task_html, get_uri
from prodigy.components.loaders import JSONL, JSON
from prodigy.components.printers import pretty_print_ner
from prodigy.components.preprocess import add_tokens
# from spacy.tokens import Span
import numpy as np
# import pickle
import prodigy
import spacy
import copy
import threading
# from prodigy.util import set_hashes
JAVASCRIPT = ""
DOC_HTML = """
<style>
    .box{
    float:left;
    margin-right:20px;
}
.clear{
    clear:both;
}
</style>
<div class="box"><iframe src="{{target_doc}}" frameborder="1" scrolling="yes" width="100%" height="512" align="left"> </iframe> </div>

<div class="box"><iframe src="{{candidate_doc}}" frameborder="1" scrolling="yes" width="100%" height="512" align="right">
    </iframe>
</div>
<div class="clear"></div>
"""


class CorefScorer:
    """
    A class for scoring coreference
    """
    def __init__(self,):
        self.scores = {}
        self.cdlm = None
        self.bert = None

    def score(self, target, candidate, method='lemma'):
        tar_cand = tuple(sorted([target['mention_id'], candidate['mention_id']]))
        if tar_cand in self.scores:
            return self.scores
        if method == 'lemma':
            target_text = target['mention_text']
            candidate_text = candidate['mention_text']

            target_lemma = target['lemma']
            candidate_lemma = candidate['lemma']

            target_sent_lemmas = target['sentence_tokens']
            candidate_sent_lemmas = candidate['sentence_tokens']

            def jc(arr1, arr2):
                return len(set.intersection(arr1, arr2)) / len(set.union(arr1, arr2))

            lemma_sim = float(target_lemma in candidate_lemma or target_lemma in candidate_text or
                              candidate_lemma in target_text)
            sent_sim = jc(set(target_sent_lemmas), set(candidate_sent_lemmas))
            similarity = np.mean([0.7 * lemma_sim, 0.3 * sent_sim])
            return similarity
        elif method == 'bert':
            if not self.bert:
                self.bert = bert_score
        elif method == 'cdlm':
            if not self.cdlm:
                self.cdlm = 1
        else:
            return 1.  # no ranking

    def precomputed(self, mention_pairs, scores):
        for p, s in zip(mention_pairs, scores):
            self.scores[tuple(sorted(p))] = s


SPACY_SPAN = 'spacy_span'


def get_clusters(dataset, working_folder):
    return []


class Clustering:
    """
    A class for running incremental human-in-the-loop clustering
    """
    def __init__(self, scorer: CorefScorer, dataset, working_folder, num_cands: int = 3):
        self.scorer = scorer
        self.clusters: List[List[Dict]] = get_clusters(dataset, working_folder)
        self.num_cands = num_cands
        self.target_task = None
        self.candidate_cluster = None
        self.found_cluster = False
        self.comparisons = 0

    def candidates(self, task):
        return [c for c in self.clusters if c[0]['topic'] == task['topic']]

    def scored_candidates(self, target_task: dict, pooling=max):
        candidates = self.candidates(target_task)
        candidate_scores = [[self.scorer.score(cand_men, target_task)
                             for cand_men in cand_mentions]
                            for cand_mentions in candidates]

        return sorted([(cand, scores) for cand, scores in zip(candidates, candidate_scores)],
                      key=lambda x: pooling(x[-1]), reverse=True)

    def merge_cluster(self):
        print('merging')
        self.candidate_cluster.append(copy.deepcopy(self.target_task))
        # self.target_task = None
        self.found_cluster = True

    def add_cluster(self):
        self.clusters.append([self.target_task])

    def make_tasks(self, examples):
        for target_task in examples:

            # set found cluster for the target to False
            self.found_cluster = False

            # get candidate tasks sorted by their max scores
            scored_candidates = self.scored_candidates(target_task)[:self.num_cands]

            for cand_tasks, scores in scored_candidates:
                # print('making candidate', len(cand_tasks))
                if self.found_cluster:
                    # print('no need to continue')
                    break

                max_score = max(scores)
                if max_score > -1:
                    # print('max_score')
                    cand_task = sorted(list(zip(cand_tasks, scores)),
                                               key=lambda x: x[-1],
                                               reverse=True)[0][0]
                    # print(cand_task)
                    coref_task = copy.deepcopy(target_task)

                    coref_task.pop('marked_doc')
                    cand_task['target_doc'] = f'http://localhost:8090/{target_task["mention_id"]}.html#{target_task["mention_id"]}'
                    cand_task['candidate_doc'] = f'http://localhost:8090/{cand_task["mention_id"]}.html#{cand_task["mention_id"]}'

                    target_task[
                        'target_doc'] = f'http://localhost:8090/{target_task["mention_id"]}.html#{target_task["mention_id"]}'
                    target_task[
                        'candidate_doc'] = f'http://localhost:8090/{cand_task["mention_id"]}.html#{cand_task["mention_id"]}'

                    self.target_task = target_task
                    self.candidate_cluster = cand_tasks
                    self.comparisons += 1

                    target_task['_task_hash'] = hash(target_task['mention_id'])
                    target_task['_input_hash'] = -hash(target_task['mention_id'])
                    yield target_task
            self.target_task = target_task
            if not self.found_cluster:
                self.add_cluster()


@prodigy.recipe(
    "evt-coref",
    dataset=("The dataset to use", "positional", None, str),
    spacy_model=("The base model", "positional", None, str),
    source=("JSON/JSONL file", "positional", None, str),
    num_cands=("No. of candidates to filter out", "option", 'n', int),
)
def event_coref_recipe(dataset: str, spacy_model: str, source: str, num_cands: int = 3,):
    """
    Create gold-standard data for events in text by updating model-in-the-loop
    """

    ##

    print(f'Using\nspacy: {spacy_model}')

    # Load the spaCy model.
    nlp = spacy.load(spacy_model)
    from parse_ecb import WhitespaceTokenizer
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    coref_scorer = CorefScorer()

    # create clustering class. Build the clusters from existing annotations
    clusterer = Clustering(coref_scorer, dataset, './tmp/', num_cands)

    labels = ['COREF']

    # load the data
    if source.lower().endswith('jsonl'):
        stream = JSONL(source)
    elif source.lower().endswith('json'):
        stream = JSON(source)
    else:
        raise TypeError("Unknown Input Format")

    tasks = list(stream)
    stream = add_tokens(nlp, tasks)
    stream = clusterer.make_tasks(stream)

    # print(stream)

    def make_update(answers):
        answer = answers[0]
        if answer['answer'] == 'accept':
            clusterer.merge_cluster()
            # time.sleep(2)
        # return 1

    update = True

    blocks = [
        {'view_id': 'ner'},
        # {"view_id": "html", "html_template": "<p>{{doc_text}}</p>"}
        {"view_id": "html", "html_template": DOC_HTML}
    ]

    config = {
        "lang": nlp.lang,
        "labels": labels,  # Selectable label options
        "span_labels": labels,  # Selectable label options
        # "exclude_by": "input",  # Hash value to filter out seen examples
        "auto_count_stream": not update,  # Whether to recount the stream at initialization
        "show_stats": True,
        "blocks": blocks,
        "custom_theme": {"cardMaxWidth": 1000},
        "feed_overlap": False,
        "javascript": JAVASCRIPT,
    }

    # config["batch_size"] = 10

    if update:
        config["batch_size"] = 1780
        config["instant_submit"] = True

    def before_db(answers):
        # print('hellp')
        return answers

    ctrl_dict = {
        "view_id": "blocks",  # Annotation interface to use
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": stream,  # Incoming stream of examples
        "update": make_update if update else None,  # Update the model in the loop if required
        "config": config,
        "before_db": before_db,
        # "on_exit": on_exit
    }
    # print(ctrl_dict['stream'])

    return ctrl_dict
    # view_id = 'cli'


def cli_annotations(ctrl_dict):
    tasks_stream = ctrl_dict['stream']
    for i, task in enumerate(tasks_stream):
        pretty_print_ner([task])
        if i == 10:
            break


if __name__ == '__main__':
    source = './ecb_test_set.json'
    lexicon = 'EMPTY'
    # lexicon = 'https://github.com/propbank/propbank-frames.git'
    spacy_model = 'en_core_web_md'

    ctrl = event_coref_recipe('', spacy_model, source, num_cands=3)

    for i, task in enumerate(ctrl['stream']):
        pretty_print_ner([task])
        ans = input('Coreferent? y/n')
        if ans == 'y':
            task['answer'] = 'accept'
            ctrl['update']([task])



