import pickle
from helper import remove_puncts, jc, generate_mention_pairs, get_topic2mention_ids
import numpy as np
from tqdm import tqdm
# from collections import defaultdict
import math
from random import random
from evaluate import load
from matplotlib import pyplot as plt


def get_mention_pair_similarity_bertscore(mention_pairs, mention_map):
    def get_b_sent(mention_map, m_id):
        return mention_map[m_id]['mention_text'] + ' [SEP] ' + mention_map[m_id]['sentence']

    m1_sentences = [get_b_sent(mention_map, m1).lower() for m1, m2 in mention_pairs]
    m2_sentences = [get_b_sent(mention_map, m2).lower() for m1, m2 in mention_pairs]

    m1_texts = [mention_map[m1]['mention_text'].lower() for m1, m2 in mention_pairs]
    m2_texts = [mention_map[m2]['mention_text'].lower() for m1, m2 in mention_pairs]

    bertscore = load("bertscore")
    pairwise_scores_sent = bertscore.compute(predictions=m1_sentences, references=m2_sentences, lang="en",
                                        model_type='distilbert-base-uncased', num_layers=6, verbose=True)

    pairwise_scores_texts = bertscore.compute(predictions=m1_texts, references=m2_texts, lang="en",
                                             model_type='distilbert-base-uncased', num_layers=6, verbose=True)

    sent_scores = np.array(pairwise_scores_sent['f1'])
    m_text_scores = np.array(pairwise_scores_texts['f1'])

    return m_text_scores, sent_scores


def get_mention_pair_similarity_cdlm(mention_pairs, mention_map):
    def get_b_sent(mention_map, m_id):
        return mention_map[m_id]['mention_text'] + ' [SEP] ' + mention_map[m_id]['sentence']

    m1_sentences = [get_b_sent(mention_map, m1) for m1, m2 in mention_pairs]
    m2_sentences = [get_b_sent(mention_map, m2) for m1, m2 in mention_pairs]

    m1_texts = [mention_map[m1]['mention_text'] for m1, m2 in mention_pairs]
    m2_texts = [mention_map[m2]['mention_text'] for m1, m2 in mention_pairs]

    bertscore = load("bertscore")
    pairwise_scores_sent = bertscore.compute(predictions=m1_sentences, references=m2_sentences, lang="en",
                                        model_type='distilbert-base-uncased', num_layers=6, verbose=True)

    pairwise_scores_texts = bertscore.compute(predictions=m1_texts, references=m2_texts, lang="en",
                                             model_type='distilbert-base-uncased', num_layers=6, verbose=True)

    sent_scores = np.array(pairwise_scores_sent['f1'])
    m_text_scores = np.array(pairwise_scores_texts['f1'])

    return m_text_scores + sent_scores





def get_mention_pair_similarity_lemma_value(mention_pairs, mention_map):
    similarities = []
    lemma_sims = []
    sent_sims = []
    # generate similarity using the mention text
    for pair in tqdm(mention_pairs, desc='Generating Similarities'):
        men1, men2 = pair
        men_map1 = mention_map[men1]
        men_map2 = mention_map[men2]
        men_text1 = remove_puncts(men_map1['mention_text'].lower())
        men_text2 = remove_puncts(men_map2['mention_text'].lower())
        lemma1 = remove_puncts(men_map1['lemma'].lower())
        lemma2 = remove_puncts(men_map2['lemma'].lower())

        # doc_id1 = men_map1['doc_id']
        # sent_id1 = int(men_map1['sentence_id'])
        # all_sent_ids1 = {str(sent_id1 - 1), str(sent_id1), str(sent_id1 + 1)}
        # all_sent_ids1 = {str(sent_id1)}
        #
        # doc_id2 = men_map2['doc_id']
        # sent_id2 = int(men_map2['sentence_id'])
        # all_sent_ids2 = {str(sent_id2 - 1), str(sent_id2), str(sent_id2 + 1)}
        #
        # all_sent_ids2 = {str(sent_id2)}

        # sentence_tokens1 = [tok for sent_id in all_sent_ids1 if sent_id in doc_sent_map[doc_id1]
        #                     for tok in doc_sent_map[doc_id1][sent_id]['sentence_tokens']]
        #
        # sentence_tokens2 = [tok for sent_id in all_sent_ids2 if sent_id in doc_sent_map[doc_id2]
        #                     for tok in doc_sent_map[doc_id2][sent_id]['sentence_tokens']]

        sentence_tokens1 = [tok.lower() for tok in men_map1['sentence_tokens']]

        sentence_tokens2 = [tok.lower() for tok in men_map2['sentence_tokens']]

        sent_sim = jc(set(sentence_tokens1), set(sentence_tokens2))
        # sent_sim = jc(set(men_map1['sentence_tokens']), set(men_map2['sentence_tokens']))
        # doc_sim = doc_sims[doc2id[men_map1['doc_id']], doc2id[men_map2['doc_id']]]
        lemma_sim = float(lemma1 in men_text2 or lemma2 in men_text1
                          or men_text1 in lemma2)
        lemma_sims.append(lemma_sim)

        sent_sims.append(sent_sim)
        # similarities.append(0.8*lemma_sim + 0.2*sent_sim)

    return np.array(lemma_sims), np.array(sent_sims)


def run_incremental_simulation(evt_mention_map_split, pair_similarity_map, top_n, threshold=0, no_sort=False):
    current_events = list(evt_mention_map_split.keys())

    # simulate by topic
    topic2mention_ids = get_topic2mention_ids(evt_mention_map_split, current_events)

    # simulation metrics
    comparisons = 0
    positive_comparisons = 0
    # negative_comparisons = 0
    total_positive_comparisons = 0

    for topic_id, topic_mention_ids in topic2mention_ids.items():
        topic_clusters = []
        for event_id in topic_mention_ids:
            cluster_similarities = []
            for clus in topic_clusters:
                clus_sim = max([pair_similarity_map[tuple(sorted([event_id, m_id]))] for m_id in clus])
                cluster_similarities.append(clus_sim)

            if no_sort:
                sorted_candidates = list(zip(topic_clusters, cluster_similarities))
            else:
                sorted_candidates = sorted(list(zip(topic_clusters, cluster_similarities)), key=lambda x: x[-1],
                                           reverse=True)

            prob_top_n = math.floor(top_n) + int(random() < (top_n - math.floor(top_n)))

            # pruning by top-n
            pruned_candidates = sorted_candidates[:prob_top_n]
            # pruning by threshold
            pruned_candidates = [c for c, sim in pruned_candidates if sim >= threshold]

            # using ground-truth
            event_true_cluster = evt_mention_map_split[event_id]['gold_cluster']
            all_candidates_true_clusters = [evt_mention_map_split[c[0]]['gold_cluster'] for c, sim in sorted_candidates]
            pruned_candidates_true_clusters = [evt_mention_map_split[c[0]]['gold_cluster'] for c in pruned_candidates]

            if event_true_cluster in pruned_candidates_true_clusters:
                positive_comparisons += 1
                total_positive_comparisons += 1
                predicted_clus_index = pruned_candidates_true_clusters.index(event_true_cluster)
                comparisons += predicted_clus_index + 1
                predicted_clus = pruned_candidates[predicted_clus_index]
                predicted_clus.append(event_id)
            else:
                comparisons += len(pruned_candidates)
                if event_true_cluster in all_candidates_true_clusters:
                    total_positive_comparisons += 1
                topic_clusters.append([event_id])

    return comparisons, positive_comparisons, total_positive_comparisons


def get_pair_sim_map(pairs, similarities):
    sim_map = {}
    for p, sim in zip(pairs, similarities):
        p_tuple = tuple(sorted(p))
        sim_map[p_tuple] = sim
    return sim_map


def run_annotation_simulation_lemma(dataset, split, ns=None, dev=False, my_lam=0.7, no_sort=False):
    if ns is None:
        # ns = [i/2 for i in range(2, 41)]
        ns = [i/2 for i in range(2, 30)]
    dataset_folder = f'./datasets/{dataset}/'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == split}

    all_mention_pairs = generate_mention_pairs(evt_mention_map, split)
    print("Total mention pairs in the Test set:", len(all_mention_pairs))

    men_sims, sent_sims = get_mention_pair_similarity_lemma_value(all_mention_pairs, evt_mention_map)
    return get_results_mention_sent(evt_mention_map, all_mention_pairs, ns, men_sims, sent_sims, my_lam, dev=dev, no_sort=no_sort)


def get_results(evt_mention_map, all_mention_pairs, similarities, ns, threshold=-1, no_sort=False):
    pair_similarity_map = get_pair_sim_map(all_mention_pairs, similarities)
    all_results = []
    for n in ns:
        comparisons, positive_comparisons, total_positive_comparisons = run_incremental_simulation(evt_mention_map,
                                                                                                   pair_similarity_map,
                                                                                                   n, threshold=threshold,
                                                                                                   no_sort=no_sort)
        recall = positive_comparisons / total_positive_comparisons
        precision = positive_comparisons / comparisons
        all_results.append((n, comparisons, recall, precision))
    return all_results


def get_results_mention_sent(evt_mention_map, all_mention_pairs, ns, men_sims, sent_sims, my_lam, dev=False, no_sort=False):
    plt.style.use('ggplot')
    if dev:
        lams = [i / 10 for i in range(1, 11)]
        # lams = [0.6 + i / 100 for i in range(1, 11)]
        # lams = [0.5, 0.6, 0.7, 0.8]
        lam_results = []
        popular_markers = ['o', 's', 'D', 'v', '^', '<', '>', 'p', 'h', '*', ',']
        # markers = ['s', 'o', '*', 'x', 'd', 'P', 'v', '<', '>', '.', '.', ',']
        fig, ax = plt.subplots(1, 1)
        for i, lam in enumerate(lams):
            curr_sims = lam * men_sims + (1 - lam) * sent_sims
            curr_lam_results = get_results(evt_mention_map, all_mention_pairs, curr_sims, ns, no_sort=no_sort)
            _, curr_comps, curr_recall, _ = zip(*curr_lam_results)

            ax.plot(curr_comps, curr_recall, '-', markersize=3, marker=popular_markers[i], label='%.1f' % lam, linewidth=0.1)

        # ax.set_yticks([(90 + i) / 100 for i in range(11)])
        # ax.set_yticklabels([str((90 + i) / 100) if i % 2 == 0 else None for i in range(11)])
        # ax.set_ylabel(r'$/mucr$', **courier)
        ax.set_ylim(0.70, 1.005)

        ax.set_xscale('log')
        ax.set_xticks([i * 1000 for i in range(2, 12)])
        ax.set_xlim(2000, 9000)
        ax.set_xticklabels([str(i) for i in range(2, 12)])
        ax.set_xlabel(r'$/comps~\times~10^3~(~\log~)$')

        ax.grid(alpha=0.5)
        ax.legend()
        plt.show()
    else:
        curr_sims = my_lam * men_sims + (1 - my_lam) * sent_sims
        return get_results(evt_mention_map, all_mention_pairs, curr_sims, ns, no_sort=no_sort)


def run_annotation_simulation_ce(dataset, split, pairs_pkl, scores_ab_pkl, scores_ba_pkl=None, ns=None, threshold=0.0):
    if ns is None:
        # ns = [i/2 for i in range(2, 41)]
        ns = [i/2 for i in range(2, 41)]

    dataset_folder = f'./datasets/{dataset}/'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == split}

    all_mention_pairs = pickle.load(open(pairs_pkl, 'rb'))
    scores_ab = pickle.load(open(scores_ab_pkl, 'rb'))
    if scores_ba_pkl:
        scores_ba = pickle.load(open(scores_ba_pkl, 'rb'))
        scores_ab = (scores_ab + scores_ba)/2

    thresholds = [i/10 for i in range(11)]
    # threshold_analysis(evt_mention_map, all_mention_pairs, ns, np.array(scores_ab), thresholds)

    return get_results(evt_mention_map, all_mention_pairs, scores_ab, ns, threshold=threshold, no_sort=False)


def run_annotation_simulation_bert_score(dataset, split, my_lam=0.69, ns=None, dev=False):
    if ns is None:
        # ns = [i/2 for i in range(2, 41)]
        ns = [i/2 for i in range(2, 41)]
    dataset_folder = f'./datasets/{dataset}/'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == split}

    all_mention_pairs = generate_mention_pairs(evt_mention_map, split)
    print("Total mention pairs in the Test set:", len(all_mention_pairs))

    # lemma_similarities = get_mention_pair_similarity_lemma_value(all_mention_pairs, evt_mention_map)
    m_text_sims, sent_sims = get_mention_pair_similarity_bertscore(all_mention_pairs, evt_mention_map)
    return get_results_mention_sent(evt_mention_map, all_mention_pairs, ns, m_text_sims, sent_sims, my_lam, dev=dev)


def threshold_analysis(evt_mention_map, all_mention_pairs, ns, similarities, thresholds=None):
    plt.style.use('ggplot')
    popular_markers = ['o', 's', 'D', 'v', '^', '<', '>', 'p', 'h', '*', ',']
    fig, ax = plt.subplots(1, 1)
    for i, thres in tqdm(enumerate(thresholds), desc='plotting'):
        curr_lam_results = get_results(evt_mention_map, all_mention_pairs, similarities, ns, threshold=thres)
        _, curr_comps, curr_recall, _ = zip(*curr_lam_results)
        ax.plot(curr_comps, curr_recall, '-', markersize=4, marker=popular_markers[i], label='%.1f' % thres,
                linewidth=0.1)

    ax.set_ylim(0.75, 1.005)

    ax.set_xscale('log')
    ax.set_xticks([i * 1000 for i in range(2, 12)])
    ax.set_xlim(700, 9000)
    ax.set_xticklabels([str(i) for i in range(2, 12)])
    ax.set_xlabel(r'$/comps~\times~10^3~(~\log~)$')

    ax.grid(alpha=0.5)
    ax.legend()
    plt.show()


def run_thres_analysis_bert_score(dataset, split, my_lam=0.69, ns=None, thresholds=None):
    if ns is None:
        # ns = [i/2 for i in range(2, 41)]
        ns = [i/2 for i in range(2, 10)]
    dataset_folder = f'./datasets/{dataset}/'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == split}

    all_mention_pairs = generate_mention_pairs(evt_mention_map, split)
    print("Total mention pairs in the Test set:", len(all_mention_pairs))

    # lemma_similarities = get_mention_pair_similarity_lemma_value(all_mention_pairs, evt_mention_map)
    m_text_sims, sent_sims = get_mention_pair_similarity_bertscore(all_mention_pairs, evt_mention_map)

    similarities = my_lam*m_text_sims + (1 - my_lam)*sent_sims

    if thresholds is None:
        thresholds = [0.7 + i/100 for i in range(0, 11)]

    threshold_analysis(evt_mention_map, all_mention_pairs, ns, similarities, thresholds)


def run_thres_analysis_lemma(dataset, split, my_lam=0.69, ns=None, thresholds=None):
    if ns is None:
        # ns = [i/2 for i in range(2, 41)]
        ns = [i/2 for i in range(2, 10)]
    dataset_folder = f'./datasets/{dataset}/'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == split}

    all_mention_pairs = generate_mention_pairs(evt_mention_map, split)
    print("Total mention pairs in the Test set:", len(all_mention_pairs))

    # lemma_similarities = get_mention_pair_similarity_lemma_value(all_mention_pairs, evt_mention_map)
    m_text_sims, sent_sims = get_mention_pair_similarity_lemma_value(all_mention_pairs, evt_mention_map)

    similarities = my_lam*m_text_sims + (1 - my_lam)*sent_sims

    if thresholds is None:
        thresholds = [i/10 for i in range(0, 11)]

    threshold_analysis(evt_mention_map, all_mention_pairs, ns, similarities, thresholds)


WIDTH = 455.24411


def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
#     print(fig_height_in)
    return fig_width_in, fig_height_in + 1


def lambda_analysis(dataset, evt_mention_map, all_mention_pairs, lams, ns, men_sims, sent_sims, ax, title, y_label):
    popular_markers = ['o', 's', 'D', 'v', '^', '<', '>', 'p', 'h', '*', ',']
    for i, lam in enumerate(lams):
        curr_sims = lam * men_sims + (1 - lam) * sent_sims
        curr_lam_results = get_results(evt_mention_map, all_mention_pairs, curr_sims, ns)
        _, curr_comps, curr_recall, _ = zip(*curr_lam_results)

        ax.plot(curr_comps, curr_recall, '-', markersize=5, marker=popular_markers[i], label='%.1f' % lam,
                linewidth=0.1)

    # ax.set_yticks([(90 + i) / 100 for i in range(11)])
    # ax.set_yticklabels([str((90 + i) / 100) if i % 2 == 0 else None for i in range(11)])
    ax.set_title(title)
    ax.set_ylabel(f'{y_label} ~ /mucr')
    ax.set_ylim(0.75, 1.005)
    ax.set_xlabel(r'$/comps~\times~10^3~(~\log~)$')
    if dataset == 'ecb':
        ax.set_xscale('log')
        ax.set_xticks([i * 1000 for i in range(2, 12)])
        ax.set_xlim(2000, 9000)
        ax.set_xticklabels([str(i) for i in range(2, 12)])
    ax.grid(alpha=0.5)
    # ax.legend()


def run_lambda_analysis(dataset, split, my_lams=None, ns=None):
    if ns is None:
        # ns = [i/2 for i in range(10, 15)]
        ns = [i for i in range(2, 20)]

    if my_lams is None:
        my_lams = [0, 0.3, 0.5, 0.7, 0.8, 1.0]
        # my_lams = [l_/10 for l_ in range(11)]

    dataset_folder = f'./datasets/ecb/'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map_ecb = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == split}

    all_mention_pairs_ecb = generate_mention_pairs(evt_mention_map_ecb, split)
    print("Total mention pairs in the Test set:", len(all_mention_pairs_ecb))

    ecb_men_sims_l, ecb_sent_sims_l = get_mention_pair_similarity_lemma_value(all_mention_pairs_ecb, evt_mention_map_ecb)
    ecb_men_sims_b, ecb_sent_sims_b = get_mention_pair_similarity_bertscore(all_mention_pairs_ecb, evt_mention_map_ecb)

    dataset_folder = f'./datasets/gvc/'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map_gvc = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == split}

    all_mention_pairs_gvc = generate_mention_pairs(evt_mention_map_gvc, split)
    print("Total mention pairs in the Test set:", len(all_mention_pairs_gvc))

    gvc_men_sims_l, gvc_sent_sims_l = get_mention_pair_similarity_lemma_value(all_mention_pairs_gvc,
                                                                              evt_mention_map_gvc)
    gvc_men_sims_b, gvc_sent_sims_b = get_mention_pair_similarity_bertscore(all_mention_pairs_gvc, evt_mention_map_gvc)

    plt.style.use('ggplot')
    plt.rcParams["pgf.texsystem"] = "pdflatex"
    # plt.rcParams['figure.figsize'] = [12, 5.5]
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 9
    plt.rcParams["font.family"] = "Times New Roman"
    fig, axes = plt.subplots(2, 2, figsize=set_size(WIDTH, fraction=1, subplots=(2, 2)), constrained_layout=True)
    lambda_analysis('ecb', evt_mention_map_ecb, all_mention_pairs_ecb, my_lams, ns, ecb_men_sims_l, ecb_sent_sims_l, axes[0, 0], title='/ecb', y_label='/lemma')
    lambda_analysis('ecb', evt_mention_map_ecb, all_mention_pairs_ecb, my_lams, ns, ecb_men_sims_b, ecb_sent_sims_b, axes[1, 0], None, y_label='/bert')

    lambda_analysis('gvc', evt_mention_map_gvc, all_mention_pairs_gvc, my_lams, ns, gvc_men_sims_l, gvc_sent_sims_l,
                    axes[0, 1], title='/gvc', y_label='/lemma')
    lambda_analysis('gvc', evt_mention_map_gvc, all_mention_pairs_gvc, my_lams, ns, gvc_men_sims_b, gvc_sent_sims_b,
                    axes[1, 1], None, y_label='/bert')

    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(loc='lower center', bbox_to_anchor=(0.57, -0.10), handles=handles, labels=labels, ncol=6, fancybox=True,
               shadow=False)
    plt.savefig('lambda-analysis.pgf', bbox_inches='tight')
    plt.show()


if __name__=='__main__':
    pairs_pk = './datasets/ecb/dpos/dev_lh_oracle_pairs_all.pkl'
    scores_ab_pk = './datasets/ecb/dpos/dev_lh_oracle_scores_ab_all.pkl'

    # print(run_annotation_simulation_ce('ecb', 'dev', pairs_pk, scores_ab_pk, dev=False))
    # run_thres_analysis_bert_score('ecb', 'dev')
    # run_thres_analysis_lemma('ecb', 'dev')
    run_lambda_analysis('ecb', 'dev',)
    run_lambda_analysis('gvc', 'dev',)