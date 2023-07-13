import copy
import os
import pickle
import sys
from collections import defaultdict, OrderedDict
import zipfile
import glob
import spacy
from bs4 import BeautifulSoup
from copy import deepcopy
from tqdm.autonotebook import tqdm
import json
from spacy.tokens import Doc
import json


VALIDATION = ['2', '5', '12', '18', '21', '23', '34', '35']
TRAIN = [str(i) for i in range(1, 36) if str(i) not in VALIDATION]
TEST = [str(i) for i in range(36, 46)]


class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        spaces = [True] * len(words)
        # Avoid zero-length tokens
        for i, word in enumerate(words):
            if word == "":
                words[i] = " "
                spaces[i] = False
        # Remove the final trailing space
        if words[-1] == " ":
            words = words[0:-1]
            spaces = spaces[0:-1]
        else:
            spaces[-1] = False

        return Doc(self.vocab, words=words, spaces=spaces)


def get_sent_map_simple(doc_bs):
    """

    Parameters
    ----------
    doc_bs: BeautifulSoup

    Returns
    -------
    dict
    """
    sent_map = OrderedDict()

    doc_name = doc_bs.find('document')['doc_name']
    topic = doc_name.split('_')[0]

    # get all tokens
    tokens = doc_bs.find_all('token')

    # all tokens map
    all_token_map = OrderedDict()

    for token in tokens:
        all_token_map[token['t_id']] = dict(token.attrs)
        all_token_map[token['t_id']]['text'] = token.text

        if token['sentence'] not in sent_map:
            sent_map[token['sentence']] = {
                'sent_id': token['sentence'],
                'topic': topic,
                'token_map': OrderedDict()
            }
        sent_map[token['sentence']]['token_map'][token['t_id']] = all_token_map[token['t_id']]

    for val in sent_map.values():
        token_map = val['token_map']
        val['sentence'] = ' '.join([m['text'] for m in token_map.values()])

    return all_token_map, sent_map


def get_split(topic):
    if topic in TRAIN:
        split = 'train'
    elif topic in VALIDATION:
        split = 'dev'
    else:
        split = 'test'
    return split


def parse_annotations(annotation_folder, output_folder, spacy_model='en_core_web_sm'):
    """
    Read the annotations files from ECB+_LREC2014

    Parameters
    ----------
    annotation_folder
    output_folder
    spacy_model

    Returns
    -------

    """
    # get validated sentences as a map {topic: {doc_name: [sentences]}}
    valid_sentences_path = os.path.join(annotation_folder, 'ECBplus_coreference_sentences.csv')
    valid_topic_sentence_map = defaultdict(dict)
    with open(valid_sentences_path) as vf:
        rows = [line.strip().split(',') for line in vf.readlines()][1:]
        for topic, doc, sentence in rows:
            doc_name = topic + '_' + doc + '.xml'
            if doc_name not in valid_topic_sentence_map[topic]:
                valid_topic_sentence_map[topic][doc_name] = set()
            valid_topic_sentence_map[topic][doc_name].add(sentence)

    # unzip ECB+.zip
    with zipfile.ZipFile(os.path.join(annotation_folder, 'ECB+.zip'), 'r') as zip_f:
        zip_f.extractall(output_folder)

    # read annotations files at working_folder/ECB+
    ecb_plus_folder = os.path.join(output_folder, 'ECB+/')
    doc_sent_map = {}
    mention_map = {}
    singleton_idx = 10000000000

    for ann_file in tqdm(list(glob.glob(ecb_plus_folder + "/*/*.xml")), desc='Reading ECB Corpus'):
        ann_bs = BeautifulSoup(open(ann_file, 'r').read(), 'lxml')
        doc_name = ann_bs.find('document')['doc_name']
        topic = doc_name.split('_')[0]
        # add document in doc_sent_map
        curr_tok_map, doc_sent_map[doc_name] = get_sent_map_simple(ann_bs)
        # get events and entities
        entities, events, instances = {}, {}, {}
        markables = [a for a in ann_bs.find('markables').children if a.name is not None]
        for mark in markables:
            if mark.find('token_anchor') is None:
                instances[mark['m_id']] = mark.attrs
            elif 'action' in mark.name or 'neg' in mark.name:
                events[mark['m_id']] = mark
            else:
                entities[mark['m_id']] = mark

        # relations
        relation_map = {}
        relations = [a for a in ann_bs.find('relations').children if a.name is not None]
        for relation in relations:
            target_m_id = relation.find('target')['m_id']
            source_m_ids = [s['m_id'] for s in relation.find_all('source')]
            for source in source_m_ids:
                relation_map[source] = target_m_id

        # create mention_map
        for m_id, mark in {**entities, **events}.items():
            if m_id in entities:
                men_type = 'ent'
            else:
                men_type = 'evt'

            split = get_split(topic)

            mention_tokens = [curr_tok_map[m['t_id']] for m in mark.find_all('token_anchor')]
            if '36_4ecbplus.xml' in doc_name and mention_tokens[-1]['t_id'] == '127':
                mention_tokens = mention_tokens[-1:]

            sent_id = mention_tokens[0]['sentence']
            if doc_name not in valid_topic_sentence_map[topic] or \
                    sent_id not in valid_topic_sentence_map[topic][doc_name]:
                continue
            mention_id = doc_name + '_' + m_id
            mention = {
                'mention_id': mention_id,
                'm_id': m_id,
                'sentence_id':  sent_id,
                'topic': topic,
                'men_type': men_type,
                'split': split,
                'mention_text': ' '.join([m['text'] for m in mention_tokens]),
                'sentence': doc_sent_map[doc_name][sent_id]['sentence'],
                'doc_id': doc_name,
                'type': mark.name
            }

            # add bert_sentence
            sent_token_map = deepcopy(doc_sent_map[doc_name][sent_id]['token_map'])
            first_token_id = mention_tokens[0]['t_id']
            final_token_id = mention_tokens[-1]['t_id']
            mention['start'] = int(sent_token_map[first_token_id]['number'])
            mention['end'] = int(sent_token_map[final_token_id]['number'])
            sent_token_map[first_token_id]['text'] = f' <mark id="mark_id"> ' + sent_token_map[first_token_id]['text']
            if final_token_id not in sent_token_map:
                print(doc_name)
            sent_token_map[final_token_id]['text'] = sent_token_map[final_token_id]['text'] + ' </mark> '
            bert_sentence = ' '.join([s['text'] for s in sent_token_map.values()])
            mention['bert_sentence'] = bert_sentence

            # add bert_doc
            doc_sent_map_copy = deepcopy(doc_sent_map[doc_name])
            doc_sent_map_copy[sent_id]['sentence'] = bert_sentence
            bert_doc = '\n'.join([s['sentence'] for s in doc_sent_map_copy.values()])
            # mention['bert_doc'] = bert_doc

            mention['bert_doc'] = '\n'.join(['<p> ' + s['sentence'] for s in doc_sent_map_copy.values()])

            # coref_id
            if m_id in relation_map:
                instance = instances[relation_map[m_id]]
                # Intra doc coref case
                if 'instance_id' not in instance:
                    instance['instance_id'] = instance['m_id']
                cluster_id = instance['instance_id']
                tag_descriptor = instance['tag_descriptor']
            else:
                cluster_id = singleton_idx
                singleton_idx += 1
                tag_descriptor = 'singleton'
            mention['gold_cluster'] = cluster_id
            mention['tag_descriptor'] = tag_descriptor

            # add into mention map
            mention_map[doc_name + '_' + m_id] = mention

    # save doc_sent_map
    pickle.dump(doc_sent_map, open(output_folder + '/doc_sent_map.pkl', 'wb'))

    # lexical features
    nlp = spacy.load(spacy_model)
    add_lexical_features(nlp, mention_map)

    # save pickle
    pickle.dump(mention_map, open(output_folder + '/mention_map.pkl', 'wb'))

    return doc_sent_map, mention_map


def add_lexical_features(nlp, mention_map):
    """
    Add lemma, derivational verb, etc
    Parameters
    ----------
    nlp: spacy.tokens.Language
    mention_map: dict

    Returns
    -------
    None
    """

    mentions = list(mention_map.values())

    mention_sentences = [mention['sentence'] for mention in mentions]
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

    for i, doc in tqdm(enumerate(nlp.pipe(mention_sentences)),
                       total=len(mention_sentences),
                       desc='Adding Lexical Features'):
        mention = mentions[i]

        # get mention span
        mention_span = doc[mention['start']:mention['end']+1]

        mention['start_char'] = mention_span.start_char
        mention['end_char'] = mention_span.end_char

        # add additional info for events
        if mention['men_type'] == 'evt':
            # add char spans of root
            root_index = mention_span.root.i
            root_span = doc[root_index:root_index+1]
            # mention['start_char'] = root_span.start_char
            # mention['end_char'] = root_span.end_char

            # get lemma
            mention['lemma'] = ' '.join([w.lemma_ for w in mention_span])
            mention['lemma_root'] = doc[root_index].lemma_

            # lemma_start and end chars
            mention['pos'] = mention_span.root.pos_

            # sentence tokens
            mention['sentence_tokens'] = [w.lemma_.lower() for w in doc if (not (w.is_stop or w.is_punct)) or
                                          w.lemma_.lower() in {'he', 'she', 'his', 'him', 'her'}]

            mention['has_pron'] = len(
                set(mention['sentence_tokens']).intersection({'he', 'she', 'his', 'him', 'her'})) > 0


def save_raw_and_tasks(ann_dir, output_folder, spacy_model='en_core_web_md'):
    doc_sent_path = output_folder + '/doc_sent_map.pkl'
    mention_map_path = output_folder + '/mention_map.pkl'
    if not os.path.exists(doc_sent_path) or not os.path.exists(mention_map_path):
        doc_sent_map, mention_map = parse_annotations(ann_dir, output_folder, spacy_model)
    else:
        doc_sent_map = pickle.load(open(doc_sent_path, 'rb'))
        mention_map = pickle.load(open(mention_map_path, 'rb'))

    sentence_mention_map = {}
    for men_id, mention in mention_map.items():
        # sentence id
        doc_id, sent_id = (mention['doc_id'], str(mention['sentence_id']))
        sentence_id = (doc_id, sent_id)

        # sentence map
        if sentence_id not in sentence_mention_map:
            sentence_mention_map[sentence_id] = []
        sentence_mention_map[sentence_id].append(men_id)

    tasks = []
    raw_docs_dir = output_folder + '/raw/'
    if not os.path.exists(raw_docs_dir):
        os.makedirs(raw_docs_dir)

    for doc, sent_map in doc_sent_map.items():
        doc_txt = raw_docs_dir + doc + '.txt'
        with open(doc_txt, 'w') as rf:
            rf.write('\n'.join([sent['sentence'] for sent in sent_map.values()]))

        for i, sent in sent_map.items():
            header = "Document-" + doc + '\n' + "Sentence-" + str(i) + ' '
            tasks.append(
                {
                    'header': header,
                    'char_offset': len(header),
                    'tok_offset': 3,
                    'text': sent['sentence'],
                    'doc_id': doc,
                    'sentence_id': i,
                    'gold': {
                        'topic': sent['topic'],
                        'spans': [
                            {
                                'token_start': mention_map[men_id]['start'],
                                'token_end': mention_map[men_id]['end'],
                                'start': mention_map[men_id]['start_char'],
                                'end': mention_map[men_id]['end_char'],
                                'label': mention_map[men_id]['men_type'].upper(),
                                'split': mention_map[men_id]['split'],
                                'type': mention_map[men_id]['type'].upper(),
                                'men_type': mention_map[men_id]['men_type'],

                            }
                            for men_id in sentence_mention_map[doc, str(i)]
                        ] if (doc, str(i)) in sentence_mention_map else []
                    },
                    'meta': {
                        'Doc': doc,
                        'Sentence': int(i),
                        # "Loc": os.path.abspath(doc_txt)
                    }
                }
            )

    for task in tasks:
        task['gold']['spans'].sort(key=lambda x: x['token_start'])

    spacy_tasks = []
    for task in tasks:
        spacy_task = copy.deepcopy(task)
        if 'gold' in task and 'spans' in task['gold']:
            spacy_task['spans'] = spacy_task['gold']['spans']
        spacy_tasks.append(spacy_task)

    return tasks, spacy_tasks
    # tasks_json_path = output_folder + '/gold_tasks.json'
    # with open(tasks_json_path, 'w', encoding='utf-8') as f:
    #     json.dump(tasks, f, ensure_ascii=False, indent=4)


def save_discrete_events(ann_dir, output_folder, spacy_model='en_core_web_md'):
    mention_map_path = output_folder + '/mention_map.pkl'
    if os.path.exists(mention_map_path):
        doc_sent_map, mention_map = parse_annotations(ann_dir, output_folder, spacy_model)
    else:
        mention_map = pickle.load(open(mention_map_path, 'rb'))

    tasks = []
    for m_id, mention in mention_map.items():
        if mention['men_type'].upper() == 'EVT':
            tasks.append(
                {
                    'text': mention['sentence'],
                    'doc_id': mention['doc_id'],
                    'sentence_id': mention['sentence_id'],
                    'spans': [{
                        'token_start': mention['start'],
                        'token_end': mention['end'],
                        'start': mention['start_char'],
                        'end': mention['end_char'],
                        'label': mention['men_type'].upper(),
                        'split': mention['split'],
                        'type': mention['type'].upper(),
                        'men_type': mention['men_type'],
                        'mention_id': mention['mention_id'],
                        'gold_cluster': mention['gold_cluster']
                    }],
                    'gold': {
                        'topic': mention['topic'],
                    },
                    'meta': {
                        'Doc': mention['doc_id'],
                        'Sentence': int(mention['sentence_id']),
                        # "Loc": os.path.abspath(doc_txt)
                    }
                }
            )


def generate_tasks(mentions):
    my_tasks = []
    for mention in mentions:
        task = {
            'text': mention['sentence'],
            'topic': mention['topic'],
            'mention_id': mention['mention_id'],
            'gold_cluster': mention['gold_cluster'],
            'doc_id': mention['doc_id'],
            'marked_doc': mention['bert_doc'],
            'mention_text': mention['mention_text'],
            'sentence_tokens': mention['sentence_tokens'],
            'lemma': mention['lemma'],
            'spans': [{
                'token_start': mention['start'],
                'token_end': mention['end'],
                'start': mention['start_char'],
                'end': mention['end_char'],
                'label': mention['men_type'].upper(),
            }],
            'meta': {
                'doc_id': mention['doc_id'],
                'sentence_id': int(mention['sentence_id']),
            },
        }
        my_tasks.append(task)
    return my_tasks


if __name__ == '__main__':
    from prodigy.components.db import connect
    # en_core_web_md test
    if len(sys.argv) != 3:
        print('usage: python parse_ecb.py spacy_model ds_split')
        sys.exit(0)
    spacy_model = sys.argv[1]
    annotation_path = "./ecbPlus/ECB+_LREC2014/"
    my_split = sys.argv[2]
    working_folder = "./ecb/"
    parse_annotations(annotation_path, working_folder, spacy_model=spacy_model)
    # exit(0)
    # my_split = 'test'
    mention_map = pickle.load(open(f'./{working_folder}/mention_map.pkl', 'rb'))
    evt_mentions_split = [men for men in mention_map.values() if men['men_type'] == 'evt' and men['split'] == my_split]
    evt_mentions_split = sorted(evt_mentions_split, key=lambda x: (x['doc_id'], int(x['sentence_id']), x['start']))
    print(my_split, 'evt count:', len(evt_mentions_split))

    tasks = generate_tasks(evt_mentions_split)

    json.dump(tasks, open(f'ecb_{my_split}_set.json', 'w'), indent=1)

