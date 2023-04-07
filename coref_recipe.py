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
# from prodigy.util import set_hashes

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
<div class="box"><iframe src="{{target_doc}}" frameborder="0" scrolling="no" width="100%" height="512" align="left"> </iframe> </div>

<div class="box"><iframe src="{{candidate_doc}}" frameborder="0" scrolling="no" width="100%" height="512" align="right">
    </iframe>
</div>
<div class="clear"></div>
"""

# JAVASCRIPT = """
# document.addEventListener('prodigyanswer', event => {
#         const { answer, task } = event.detail
#         var marked_event = document.getElementById(task.mention_id);
#         marked_event.scrollIntoView({
#           behavior: "smooth",
#           block: "start",
#           inline: "nearest"
#         });
#     })
#
# function doThis() {
#         window.prodigy.update({});
#     }
# """


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
        return self.clusters

    def scored_candidates(self, target_task: dict, pooling=max):
        candidates = self.candidates(target_task)
        candidate_scores = [[self.scorer.score(cand_men, target_task)
                             for cand_men in cand_mentions]
                            for cand_mentions in candidates]

        return sorted([(cand, scores) for cand, scores in zip(candidates, candidate_scores)],
                      key=lambda x: pooling(x[-1]), reverse=True)

    def merge_cluster(self):
        print('merging')
        # print(self.target_task['mention_id'])
        self.candidate_cluster.append(self.target_task)
        # self.target_task = None
        self.found_cluster = True

    def add_cluster(self):
        self.clusters.append([self.target_task])

    def create_target_span_task(self, target_task, candidate_tasks):
        target_spacy_span = target_task.pop(SPACY_SPAN)
        target_task_pro = copy.deepcopy(target_task)
        target_task[SPACY_SPAN] = target_spacy_span
        target_span = copy.deepcopy(target_task_pro['span'])

        target_style = {"color": "blue", "background": "cyan",
                        "font": "bold 1.0em ''"}

        # the header for the Target
        target_head_tokens = [
            {'text': 'Target', 'style': target_style},
            {'text': '\n'}
        ]

        # Join the tokens to create the text header
        task_text = ''.join([tok['text'] for tok in target_head_tokens])

        def add_offsets(span, tok_offset, char_offset):
            span['token_start'] += tok_offset
            span['token_end'] += tok_offset

            # add char offsets
            span['start'] += char_offset
            span['end'] += char_offset

        task_tok_start = len(target_head_tokens)
        task_char_start = len(task_text)
        add_offsets(target_span, task_tok_start, task_char_start)

        cand_style = {"color": "green", "background": "lightgreen",
                      "font": "bold 1.0em ''"}
        cand_head_tokens = [
            {'text': '\n'},
            {'text': 'Candidate_Cluster', 'style': cand_style},
            {'text': '\n'}
        ]
        cand_header = ''.join([tok['text'] for tok in cand_head_tokens])
        task_tokens = target_head_tokens + target_task['tokens'] + cand_head_tokens
        task_text = task_text + target_task_pro['text'] + cand_header
        task_char_start = len(task_text)
        task_tok_start = len(task_tokens)

        task_spans = [target_span]
        candidate_tasks_pro = []
        for cand_task in candidate_tasks:
            new_line = {'text': '\n'}
            cand_spacy_span = cand_task.pop(SPACY_SPAN)
            cand_task_pro = copy.deepcopy(cand_task)
            cand_task[SPACY_SPAN] = cand_spacy_span
            cand_span = copy.deepcopy(cand_task_pro['span'])
            candidate_tasks_pro.append(cand_task_pro)
            add_offsets(cand_span, task_tok_start, task_char_start)

            task_tokens += cand_task_pro['tokens']
            task_tokens += [new_line]
            task_text = task_text + cand_task_pro['text'] + new_line['text']

            task_tok_start = len(task_tokens)
            task_char_start = len(task_text)

            task_spans.append(cand_span)

        return {
            'text': task_text,
            'span_pair': {
                'target': target_task_pro,
                'candidate': candidate_tasks_pro
            },
            'spans': task_spans,
            'tokens': task_tokens,
            'meta': {
                'target_doc': target_task_pro['doc_id'],
                'candidate_docs': set([task['doc_id'] for task in candidate_tasks_pro])
            }
        }

    def make_tasks(self, examples):
        for target_task in examples:
            print('making', target_task['mention_id'])
            # set found cluster for the target to False
            self.found_cluster = False

            # get candidate tasks sorted by their max scores
            scored_candidates = self.scored_candidates(target_task)[:self.num_cands]

            for cand_tasks, scores in scored_candidates:
                print('making candidate', len(cand_tasks))
                if self.found_cluster:
                    # print('no need to continue')
                    break

                max_score = max(scores)
                if max_score > 0:
                    cand_task = sorted(list(zip(cand_tasks, scores)),
                                               key=lambda x: x[-1],
                                               reverse=True)[0][0]
                    # print(cand_task)
                    coref_task = copy.deepcopy(target_task)

                    coref_task.pop('marked_doc')
                    coref_task['target_doc'] = f'http://localhost:8090/{target_task["mention_id"]}.html#{target_task["mention_id"]}'
                    coref_task['candidate_doc'] = f'http://localhost:8090/{cand_task["mention_id"]}.html#{cand_task["mention_id"]}'

                    self.target_task = target_task
                    self.candidate_cluster = cand_tasks
                    self.comparisons += 1
                    # coref_task = set_hashes(coref_task)
                    yield coref_task
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

    import subprocess
    # start a simple http server to access the generated html files
    # command = 'python3 -m http.server -b 127.0.0.1 8000'
    # p = subprocess.Popen(
    #     [command],
    #     shell=True,
    #     stdin=None,
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE,
    #     close_fds=True)
    #
    # def on_exit(varsa):
    #     p.terminate()

    labels = ['COREF']

    # load the data
    if source.lower().endswith('jsonl'):
        stream = JSONL(source)
    elif source.lower().endswith('json'):
        stream = JSON(source)
    else:
        raise TypeError("Unknown Input Format")

    tasks = list(stream)
    tasks = tasks[10:15]
    stream = add_tokens(nlp, tasks)
    # stream = all_spans(stream)
    stream = list(stream)
    stream = clusterer.make_tasks(stream)
    print(stream)

    def make_update(answers):
        print('updating')
        answer = answers[0]
        if answer['answer'] == 'accept':
            clusterer.merge_cluster()
        print(answer)

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
        # "javascript": JAVASCRIPT,
    }

    config["batch_size"] = 10

    if update:
        config["batch_size"] = 1
        config["instant_submit"] = True

    def before_db(answers):
        print('hellp')
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



