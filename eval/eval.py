"""
This file is part of FLORA licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
Portions of this file are adapted from the original FLORA implementation by Yiwen Peng, Thomas Bonald, and Fabian Suchanek, licensed under the same license.

Description: Evaluation helpers for OpenEA, DBP15K, OAEI, DBP1M, DBpedia-YAGO alignment outputs, including result loading, post-processing, and metric computation.
"""

import os
import math
import bz2
import re
import scipy.stats as st
import xml.etree.ElementTree as ET
from collections import defaultdict
from urllib.parse import unquote

DBPEDIA_RESOURCE_PREFIX = 'http://dbpedia.org/resource/'
YAGO_RESOURCE_PREFIX = 'http://yago-knowledge.org/resource/'
WIKIPAGE_REDIRECTS = 'http://dbpedia.org/ontology/wikiPageRedirects'
OWL_SAME_AS = 'http://www.w3.org/2002/07/owl#sameAs'


def ranked_candidates(candidates):
    """
    Rank candidates deterministically: score desc, then target URI, to ensure reproducibility.
    """
    return sorted(candidates.items(), key=lambda item: (-item[1], item[0]))


def load_dbpedia_redirects(redirect_path):
    """
    Load DBpedia wikiPageRedirects mappings.
    """
    redirects = {}
    if redirect_path is None or not os.path.exists(redirect_path):
        return redirects

    open_func = bz2.open if redirect_path.endswith('.bz2') else open
    with open_func(redirect_path, 'rt', encoding='UTF-8', errors='replace') as file:
        for line in file:
            terms = line.strip().split()
            if len(terms) < 3:
                continue
            subject = terms[0].strip('<>')
            predicate = terms[1].strip('<>')
            target = terms[2].strip('<>')
            if predicate == WIKIPAGE_REDIRECTS and \
                    subject.startswith(DBPEDIA_RESOURCE_PREFIX) and \
                    target.startswith(DBPEDIA_RESOURCE_PREFIX):
                redirects[subject] = target
    return redirects


def load_dbpedia_yago_results(pred_path, gold_path, threshold=0.5,
                              redirect_path=None, redirects=None, use_redirects=True):
    """
    Load DBpedia-YAGO gold pairs and FLORA predictions in DBpedia -> YAGO direction.

    Parameters
    ----------
    pred_path : str
        FLORA result TTL file with scored owl:sameAs lines.
    gold_path : str
        Gold alignment TTL file.
    threshold : float
        Only prediction pairs with score > threshold are loaded.
    redirect_path : str
        DBpedia wikiPageRedirects TTL/Turtle(.bz2) file.
    redirects : dict
        Optional DBpedia redirect map.
    use_redirects : bool
        Canonicalize DBpedia URIs by redirects. Defaults to True.

    Returns
    -------
    gold : dict
        Canonical DBpedia URI -> YAGO URI gold pairs.
    sameAsscores : dict
        Canonical DBpedia URI -> {YAGO URI: score} prediction scores.
    info : dict
        Loading statistics, including excluded conflicting gold sources.
    """
    prefixes = {
        'yago': YAGO_RESOURCE_PREFIX,
        'dbr': DBPEDIA_RESOURCE_PREFIX,
        'owl': 'http://www.w3.org/2002/07/owl#',
    }
    # DBpedia gold links and FLORA predictions may use different URIs for the same entity
    # when one side points to a redirected page. Canonicalizing DBpedia resources before
    # comparison prevents these redirect aliases from being counted as false errors.
    if redirects is not None:
        redirect_map = redirects
    elif use_redirects and redirect_path is not None:
        redirect_map = load_dbpedia_redirects(redirect_path)
    elif use_redirects:
        raise ValueError('redirect_path or redirects must be provided when use_redirects=True')
    else:
        redirect_map = {}

    def normalize_token(token, active_prefixes):
        """Normalize a token by stripping whitespace, removing trailing periods, and expanding prefixes."""
        token = token.strip()
        if token.endswith('.'):
            token = token[:-1].strip()
        if token.startswith('<') and token.endswith('>'):
            return token[1:-1]
        if ':' in token and not re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', token):
            prefix, suffix = token.split(':', 1)
            if prefix in active_prefixes:
                return active_prefixes[prefix] + suffix
        return token

    def canonicalize_dbpedia(uri):
        """Canonicalize a DBpedia URI by following redirects until a final target is reached."""
        seen = set()
        current = uri
        while current in redirect_map and current not in seen:
            seen.add(current)
            current = redirect_map[current]
        return current

    def parse_pair(left, right):
        if left.startswith(DBPEDIA_RESOURCE_PREFIX) and right.startswith(YAGO_RESOURCE_PREFIX):
            return canonicalize_dbpedia(left), right
        if right.startswith(DBPEDIA_RESOURCE_PREFIX) and left.startswith(YAGO_RESOURCE_PREFIX):
            return canonicalize_dbpedia(right), left
        return None

    def iter_ttl(path):
        local_prefixes = prefixes.copy()
        open_func = bz2.open if str(path).endswith('.bz2') else open
        with open_func(path, 'rt', encoding='UTF-8', errors='replace') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                prefix_match = re.match(r'\s*@prefix\s+([^:\s]+):\s*<([^>]+)>\s*\.\s*$', line)
                if prefix_match:
                    local_prefixes[prefix_match.group(1)] = prefix_match.group(2)
                    continue
                terms = line.split()
                if len(terms) < 3:
                    continue
                subject = normalize_token(terms[0], local_prefixes)
                predicate = normalize_token(terms[1], local_prefixes)
                obj = normalize_token(terms[2], local_prefixes)
                score = None
                if len(terms) >= 5:
                    try:
                        score = float(terms[4])
                    except ValueError:
                        score = None
                yield subject, predicate, obj, score

    raw_gold = defaultdict(set)
    for subject, predicate, obj, _score in iter_ttl(gold_path):
        if predicate not in {'owl:sameAs', OWL_SAME_AS}:
            continue
        pair = parse_pair(subject, obj)
        if pair is None:
            continue
        dbpedia_uri, yago_uri = pair
        raw_gold[dbpedia_uri].add(yago_uri)

    gold_conflicts = {source: targets for source, targets in raw_gold.items() if len(targets) > 1} # conflicting gold sources with multiple targets
    gold = {source: next(iter(targets)) for source, targets in raw_gold.items() if source not in gold_conflicts}

    sameAsscores = {}
    for subject, predicate, obj, score in iter_ttl(pred_path):
        if predicate not in {'owl:sameAs', OWL_SAME_AS}:
            continue
        if score is None:
            continue
        if score <= threshold:
            continue
        pair = parse_pair(subject, obj)
        if pair is None:
            continue
        dbpedia_uri, yago_uri = pair
        if dbpedia_uri in gold_conflicts: # skip conflicting gold sources
            continue
        sameAsscores.setdefault(dbpedia_uri, {})
        sameAsscores[dbpedia_uri][yago_uri] = max(score, sameAsscores[dbpedia_uri].get(yago_uri, float('-inf')))
    return gold, sameAsscores

def dbpedia_yago_eval(gold, sameAsscores, threshold=0.0, hits=(1, 10)):
    """
    Evaluate DBpedia-YAGO entity alignment results with a fixed score threshold.
    """
    def resource_name(uri):
        name = uri.rsplit('/', 1)[-1].rsplit('#', 1)[-1]
        name = unquote(name)
        return re.sub(r'_u([0-9A-Fa-f]{4})_', lambda match: chr(int(match.group(1), 16)), name)

    def rank_for_source(source, candidates):
        source_name = resource_name(source)
        return sorted(((target, float(score)) for target, score in candidates.items() if float(score) > threshold),
            key=lambda item: (-item[1], 0 if resource_name(item[0]) == source_name else 1, item[0])) # sort by score desc, then exact name match, then target URI

    top1_predictions = {}
    hit_counts = {k: 0 for k in hits}
    reciprocal_rank_sum = 0.0
    gold_sources_with_candidates = 0

    for source, candidates in sameAsscores.items():
        ranked = rank_for_source(source, candidates)
        if ranked:
            top1_predictions[source] = ranked[0]

    for source, gold_target in gold.items():
        ranked = rank_for_source(source, sameAsscores.get(source, {}))
        if not ranked:
            continue
        gold_sources_with_candidates += 1
        for rank, (target, _score) in enumerate(ranked, start=1):
            if target == gold_target:
                reciprocal_rank_sum += 1 / rank
                for k in hits:
                    if rank <= k:
                        hit_counts[k] += 1
                break

    tp = sum(1 for source, (target, _score) in top1_predictions.items() if gold.get(source) == target)
    fp = len(top1_predictions) - tp
    fn = len(gold) - tp
    fp_source_absent = sum(1 for source in top1_predictions if source not in gold) # predicted source not in gold
    fp_wrong_target = fp - fp_source_absent # predicted source in gold but wrong target

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    metrics = {
        'threshold': threshold,
        'gold_pairs': len(gold),
        'prediction_sources': len(top1_predictions),
        'prediction_pairs': sum(1 for candidates in sameAsscores.values() for score in candidates.values() if float(score) > threshold),
        'gold_source_coverage': gold_sources_with_candidates / len(gold) if gold else 0,
        'tp': tp,
        'fp': fp,
        'fp_source_absent_from_gold': fp_source_absent,
        'fp_wrong_target_for_gold_source': fp_wrong_target,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mrr': reciprocal_rank_sum / len(gold) if gold else 0,
    }
    for k in hits:
        metrics[f'hit@{k}'] = hit_counts[k] / len(gold) if gold else 0

    print('Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(precision, recall, f1))
    print('Hit@1: {:.4f}, Hit@10: {:.4f}, MRR: {:.4f}'.format(metrics.get('hit@1', 0), metrics.get('hit@10', 0), metrics['mrr']))

    return metrics


def load_openea_ref(loc):
    gt_pairs = []
    with open(os.path.join(loc, 'ent_links'), 'r', encoding='UTF-8') as f:
        for line in f:
            head, tail = line.strip().split('\t')
            gt_pairs.append((head, tail))
    return gt_pairs


def openea_eval(maxAssignment, y_gold, save_path=None):
    y_pred = set()
    for e1 in sorted(maxAssignment):
        # if e1.startswith('dbr:'):
        if e1.startswith('http://dbpedia.org/resource/'):
            ranked = ranked_candidates(maxAssignment[e1])
            if ranked:
                y_pred.add(tuple([e1, ranked[0][0]])) # only select the first one
    
    # calculate precision, recall, f1
    tp = len(y_gold.intersection(y_pred))
    fp = len(y_pred - y_gold)
    fn = len(y_gold) - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1: {f1:.4f}')
    if save_path is not None:
        with open(save_path, 'w') as f:
            for e1, e2 in sorted(y_pred):
                f.write(f'{e1}\t{e2}\t{maxAssignment[e1][e2]}\n')
        print(f'\nSaved final results to "{save_path}"')


def load_ent_results(file_path, prefix, threshold=0.0):
    """
    Load the entity alignment results from the given file path.
    Only consider the entities starting with the given prefix and with scores above the threshold.

    Parameters
    ----------
    file_path : str
        The path to the file containing all the entity alignment results.
    prefix : str
        The prefix to filter the entities.
    threshold : float
        The score threshold to filter the alignments.
    
    Returns
    -------
    sameAsscores : dict
        A dictionary of valid entity alignment scores.
    ent_max_assign : dict
        A dictionary of bilateral max assignment results.
    """
    # store as sameAsscores
    sameAsscores = {}
    with open(file_path, 'r') as file:
        for line in file:
            terms = line.strip().split('\t')
            if len(terms) > 4 and terms[0].startswith(prefix):
                score = float(terms[4])
                if score <= threshold:
                    continue
                e1 = terms[0]
                e2 = terms[2]
                if e1 not in sameAsscores:
                    sameAsscores[e1] = {}
                sameAsscores[e1][e2] = score
    ent_max_assign = bilateral_max_assign(sameAsscores)
    return sameAsscores, ent_max_assign


def bilateral_max_assign(sameASscore):
    match_e1_to_e2, match_e2_to_e1 = {}, {}
    for e1 in sorted(sameASscore):
        matches = sameASscore[e1]
        if matches:
            max_score = max(matches.values())
            for e2, score in ranked_candidates(matches):
                if score == max_score:
                    if e1 not in match_e1_to_e2:
                        match_e1_to_e2[e1] = {}
                    match_e1_to_e2[e1][e2] = score
                    if e2 not in match_e2_to_e1:
                        match_e2_to_e1[e2] = {}
                        match_e2_to_e1[e2][e1] = score
                        continue

                    max_score_e2 = max(match_e2_to_e1[e2].values())
                    if score > max_score_e2:
                        match_e2_to_e1[e2] = {e1: score}
                    elif max_score_e2 == score:
                        match_e2_to_e1[e2][e1] = score
    res_max_assign = {} # bilateral max assignment
    for e2 in sorted(match_e2_to_e1):
        # exact match case, avoid duplicates
        if e2 in match_e2_to_e1[e2]:
            res_max_assign[e2] = {e2: match_e2_to_e1[e2][e2]}
            continue
        for e1, _score in ranked_candidates(match_e2_to_e1[e2]):
            if e1 in match_e1_to_e2 and e2 in match_e1_to_e2.get(e1, {}):
                if e2 not in res_max_assign:
                    res_max_assign[e2] = {}
                res_max_assign[e2][e1] = match_e1_to_e2[e1][e2]
                if e1 not in res_max_assign:
                    res_max_assign[e1] = {}
                res_max_assign[e1][e2] = match_e2_to_e1[e2][e1]
    return res_max_assign


def load_dbp15k_ref(loc):
    id2ent1, id2ent2 = {}, {}
    for i in range(2):
        id2ent = {}
        with open(loc+'ent_ids_{}'.format(i+1), encoding='UTF-8') as f:
            for line in f.readlines():
                ids, ent = line.strip().split('\t')
                id2ent[int(ids)] = ent
        if i == 0:
            id2ent1 = id2ent.copy()
        else:
            id2ent2 = id2ent.copy()
    # load supervised data
    seed_pairs = {} # 30%
    with open(loc+'sup_pairs', encoding='UTF-8') as f:
        for line in f.readlines():
            e1, e2 = line.strip().split('\t')
            seed_pairs[id2ent1[int(e1)]] = id2ent2[int(e2)]
    # ref pairs
    ref_pairs = {} # test pairs (70%)
    with open(loc+'ref_pairs', encoding='UTF-8') as f:
        for line in f.readlines():
            e1, e2 = line.strip().split('\t')
            ref_pairs[id2ent1[int(e1)]] = id2ent2[int(e2)]
    return seed_pairs, ref_pairs


def dbp15k_eval(ref_pairs, sameAsscores):
    hit1, hit10, mrr = 0, 0, 0
    for k, v in ref_pairs.items():
        if k in sameAsscores and v in sameAsscores[k]:
            rank = sorted(sameAsscores[k].items(), key=lambda x: x[1], reverse=True)
            max_score = rank[0][1]
            for i, (item, score) in enumerate(rank):
                if item == v:
                    mrr += 1 / (i + 1)
                    if score == max_score:
                        hit1 += 1
                        hit10 += 1
                        break
                    if i < 10:
                        hit10 += 1
                if i >= 10:
                    break
    print(f'Hit@1: {hit1 / len(ref_pairs):.4f}')
    print(f'Hit@10: {hit10 / len(ref_pairs):.4f}')
    print(f'MRR: {mrr / len(ref_pairs):.4f}')


def load_dbp1m_ref(path):
    """Load a two-column DBP1M train/test links file."""
    ref_pairs = {}
    with open(path, encoding='UTF-8', errors='replace') as file:
        for line in file:
            terms = line.rstrip('\n').split('\t')
            if len(terms) >= 2:
                ref_pairs[terms[0].strip().strip('<>')] = terms[1].strip().strip('<>')
    return ref_pairs


def load_dbp1m_results(path, ref_pairs, threshold=0.0):
    """Load scored DBP1M owl:sameAs results in the gold source direction."""
    sameAsscores = {}
    sources = set(ref_pairs)
    with open(path, encoding='UTF-8', errors='replace') as file:
        for line in file:
            terms = line.rstrip('\n').split('\t')
            if len(terms) < 5 or terms[1].strip().strip('<>') not in {'owl:sameAs', OWL_SAME_AS}:
                continue
            try:
                score = float(terms[4])
            except ValueError:
                continue
            if score <= threshold:
                continue
            left, right = (term.strip().strip('<>') for term in (terms[0], terms[2]))
            if left in sources:
                source, target = left, right
            elif right in sources:
                source, target = right, left
            else:
                continue
            candidates = sameAsscores.setdefault(source, {})
            candidates[target] = max(score, candidates.get(target, float('-inf')))
    return sameAsscores

def dbp1m_eval(ref_pairs, sameAsscores):
    if not ref_pairs:
        raise ValueError('ref_pairs must not be empty')
    hit1 = hit10 = mrr = 0.0
    for source, target in ref_pairs.items():
        for rank, (candidate, _score) in enumerate(
                ranked_candidates(sameAsscores.get(source, {})), start=1):
            if candidate == target:
                hit1 += rank == 1
                hit10 += rank <= 10
                mrr += 1 / rank
                break
    metrics = {'hit@1': hit1 / len(ref_pairs),
               'hit@10': hit10 / len(ref_pairs),
               'mrr': mrr / len(ref_pairs)}
    print(f"Hit@1: {metrics['hit@1']:.4f}")
    print(f"Hit@10: {metrics['hit@10']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    return metrics


def confidence_interval(p, n, confidence=0.95):
    """
    Calculate the confidence interval for a given dataset.
    """
    if n == 0:
        raise ValueError("Sample size n must be greater than 0")
    
    z = st.norm.ppf((1 + confidence) / 2.)
    se = math.sqrt((p * (1 - p)) / n)
    margin = z * se
    return p, max(0, p - margin), min(1, p + margin)



def load_oaei_ref(loc, prefix):
    tree = ET.parse(os.path.join(loc, 'reference.xml'))
    root = tree.getroot()
    ns = {
        'ns': 'http://knowledgeweb.semanticweb.org/heterogeneity/alignment',
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
    }
    cells = root.findall('.//ns:map/ns:Cell', ns)

    class_gt = dict()
    property_gt = dict()
    instance_gt = dict()
    for cell in cells:
        e1 = cell.find('ns:entity1', ns).attrib['{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource']
        e2 = cell.find('ns:entity2', ns).attrib['{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource']
        if e1.startswith(prefix+'class/'):
            class_gt['<'+e1+'>'] = '<'+e2+'>'
        elif e1.startswith(prefix+'property/'):
            property_gt['<'+e1+'>'] = '<'+e2+'>'
        elif e1.startswith(prefix+'resource/'):
            instance_gt['<'+e1+'>'] = '<'+e2+'>'
        else:
            raise ValueError('Unknown type of entity: {}'.format(e1))
    return class_gt, property_gt, instance_gt


def load_full_results_oaei_kg_track(cls_gt, inst_gt, rel_gt, prefix, loc):
    """
    Load the full results from the OAEI KG track output file.
    
    Parameters
    ----------
    cls_gt : dict
        The ground truth class alignments.
    inst_gt : dict
        The ground truth instance alignments.
    rel_gt : dict
        The ground truth property alignments.
    prefix : str
        The prefix to filter the entities.
    loc : str
        The path to the output file.
    
    Returns
    -------
    y_pred_inst : dict
        The predicted instance alignments with scores.
    y_pred_class : dict
        The predicted class alignments with scores.
    y_pred_similar : dict
        The predicted similar property alignments with scores.
    y_prop_sameAs : dict
        The predicted equivalent property alignments with scores.
    """
    # a simple version -> only gold standard are considered
    # as dedicated in https://oaei.ontologymatching.org/2024/results/knowledgegraph/index.html
    y_pred_inst = dict()
    y_pred_class = dict()
    y_pred_subproperty = dict()
    y_prop_sameAs = dict()
    with open(loc, 'r') as file:
        for line in file:
            terms = line.strip().split('\t')
            if len(terms) > 4:
                if terms[0] in inst_gt: # instance
                    if terms[0] not in y_pred_inst:
                        y_pred_inst[terms[0]] = {}
                    y_pred_inst[terms[0]][terms[2]] = terms[4]
                elif terms[0] in cls_gt: # class
                    if terms[0] not in y_pred_class:
                        y_pred_class[terms[0]] = {}
                    y_pred_class[terms[0]][terms[2]] = terms[4]
                elif terms[0].startswith('<'+prefix+'property/') or \
                        terms[2].startswith('<'+prefix+'property/'): # bilateral subrelations
                    if terms[0] in rel_gt or \
                            terms[2] in rel_gt:
                        if terms[1] == 'owl:sameAs' and terms[0] in rel_gt:
                            if terms[0] not in y_prop_sameAs:
                                y_prop_sameAs[terms[0]] = {}
                            y_prop_sameAs[terms[0]][terms[2]] = terms[4]
                            continue
                        # now terms[2] in rel_gt / terms[1] not sameAs
                        if terms[0] not in y_pred_subproperty:
                            y_pred_subproperty[terms[0]] = {}
                        y_pred_subproperty[terms[0]][terms[2]] = terms[4]
    # similar relations from subrelations
    # refer to page 9 in paper for relation equations r\cong r'
    y_pred_similar = dict()
    for k in y_pred_subproperty:
        for v in y_pred_subproperty[k]:
            if k not in y_pred_similar:
                y_pred_similar[k] = {}
            y_pred_similar[k][v] = min(float(y_pred_subproperty[k][v]), float(y_pred_subproperty.get(v, {}).get(k, 1)))
            if v not in y_pred_similar:
                y_pred_similar[v] = {}
            y_pred_similar[v][k] = min(float(y_pred_subproperty[k][v]), float(y_pred_subproperty.get(v, {}).get(k, 1)))
    return y_pred_inst, y_pred_class, y_pred_similar, y_prop_sameAs



def post_process_oaei_relation_results(prefix1, prefix2, rel_pred_same, rel_pred_similar, threshold=0.1):
    """
    Post-process the relation alignment results from the OAEI KG track.

    Parameters
    ----------
    prefix1 : str
        The prefix of the source knowledge base.
    prefix2 : str
        The prefix of the target knowledge base.
    rel_pred_same : dict
        The predicted equivalent property alignments with scores.
    rel_pred_similar : dict
        The predicted similar property alignments with scores.
    threshold : float
        The score threshold to filter the alignments.
    
    Returns
    -------
    y_pred_property_post : dict
        The post-processed equivalent property alignments with scores.
    """
    # Prioritize the sameAs matches
    y_pred_property_post = {}
    for k, preds in rel_pred_same.items():
        candidates = [
            (uri, float(score)) for uri, score in preds.items()
            if uri.startswith('<' + prefix2 + 'property/')
        ]
        if not candidates:
            continue
        uri, score = sorted(candidates, key=lambda item: (-item[1], item[0]))[0]
        if score > threshold:
            y_pred_property_post.setdefault(k, {})[uri] = score
    # Find more equivalent properties from subrelations (quasi equivalence r\cong r')
    for pred in rel_pred_similar:
        if pred in y_pred_property_post: # sameAs match
            continue
        if pred.startswith('<' + prefix1 + 'property/'):
            for v in rel_pred_similar[pred]:
                if v in y_pred_property_post: # sameAs match
                    continue
                if v.startswith('<' + prefix2 + 'property/'):
                    if pred not in y_pred_property_post:
                        y_pred_property_post[pred] = {}
                    y_pred_property_post[pred][v] = float(rel_pred_similar[pred][v])
    return y_pred_property_post


def post_process_oaei_results(cls_gt, inst_gt, rel_gt,
                              cls_pred, inst_pred, rel_pred_same, rel_pred_similar,
                              prefix1, prefix2, threshold=0.1):
    # Instances
    instAlign = {}
    for k, v in inst_gt.items():
        if k not in inst_pred:
            continue
        candidates = [
            (uri, float(score)) for uri, score in inst_pred[k].items()
            if uri.startswith('<' + prefix2 + 'resource/')
        ]
        if candidates:
            candidates.sort(key=lambda item: (-item[1], item[0]))
            pred, score = candidates[0]
            for ent in candidates:
                if ent[1] < score:
                    break
                # Multiple candidates, select the exact match one if exists
                if ent[0].split('/')[-1] == k.split('/')[-1]:
                    pred = ent[0]
                    score = ent[1]
                    break
            # 1-to-1 constraint check
            if score > instAlign.get(pred, {}).get(k, 0):
                instAlign[pred] = {k: score}

    # Classes
    clsAlign = {}
    for k, v in cls_gt.items():
        if k not in cls_pred:
            continue
        candidates = [
            item for item in cls_pred[k].items()
            if item[0].startswith('<' + prefix2 + 'class/')
        ]
        if not candidates:
            continue
        # Select the highest-scoring target class deterministically.
        pred, score = sorted(candidates, key=lambda item: (-float(item[1]), item[0]))[0]
        # 1-to-1 constraint check
        if float(score) > clsAlign.get(pred, {}).get(k, 0):
            clsAlign[pred] = {k: float(score)}
        
    # Properties
    y_pred_property_post = post_process_oaei_relation_results(prefix1, prefix2, rel_pred_same, rel_pred_similar, threshold)
    relAlign = {}
    for k, v in rel_gt.items():
        if k not in y_pred_property_post:
            continue
        # the one with maximum score: choose only the maximally assigned one
        pred, score = sorted(y_pred_property_post[k].items(), key=lambda x: x[1], reverse=True)[0]
        # 1-to-1 constraint check
        if float(score) > relAlign.get(pred, {}).get(k, 0):
            relAlign[pred] = {k: float(score)}
    instAlign = {
        source: (target, score)
        for target, sources in instAlign.items()
        for source, score in sources.items()
    }
    clsAlign = {
        source: (target, score)
        for target, sources in clsAlign.items()
        for source, score in sources.items()
    }
    relAlign = {
        source: (target, score)
        for target, sources in relAlign.items()
        for source, score in sources.items()
    }
    return instAlign, clsAlign, relAlign


def oaei_kg_eval(cls_gt, inst_gt, rel_gt, 
                 cls_pred, inst_pred, rel_pred_same, rel_pred_similar,
                 prefix1, prefix2, threshold=0.1, save_path=None):
    """
    Evaluate the OAEI KG track results.
    """
    # Post-process the results
    instAlign, clsAlign, relAlign = post_process_oaei_results(
        cls_gt, inst_gt, rel_gt,
        cls_pred, inst_pred, rel_pred_same, rel_pred_similar,
        prefix1, prefix2, threshold
    )

    tp_total, fp_total, fn_total = 0, 0, 0
    final_results = {"instances": dict(), "classes": dict(), "properties": dict()}
    # Instances
    tp, fp, fn = 0, 0, 0
    for k, v in inst_gt.items():
        if k not in instAlign:
            fn += 1
            continue
        pred, score = instAlign[k]
        if float(score) <= threshold:
            fn += 1
            continue
        final_results["instances"][k] = (pred, score)
        if pred == v:
            tp += 1
        else:
            fp += 1
            fn += 1

    tp_total += tp
    fp_total += fp
    fn_total += fn
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    print('**Instances: \n   Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(precision, recall, f1))

    # Classes
    tp, fp, fn = 0, 0, 0
    for k, v in cls_gt.items():
        if k not in clsAlign:
            fn += 1
            continue
        pred, score = clsAlign[k]
        if float(score) <= threshold:
            fn += 1
            continue
        final_results["classes"][k] = (pred, score)
        if pred == v:
            tp += 1
        else:
            #print(f'Class {k} predicted as {pred} with score {score}')
            fp += 1
            fn += 1
    tp_total += tp
    fp_total += fp
    fn_total += fn
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    print('**Classes: \n   Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(precision, recall, f1))

    # Properties
    tp, fp, fn = 0, 0, 0
    for k, v in rel_gt.items():
        if k not in relAlign:
            fn += 1
            continue
        pred, score = relAlign[k]
        if score <= threshold:
            fn += 1
            continue
        final_results["properties"][k] = (pred, score)
        if pred == v:
            tp += 1
        else:
            fp += 1
            fn += 1
    # evaluate overall
    tp_total += tp
    fp_total += fp
    fn_total += fn
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    print('**Properties: \n   Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(precision, recall, f1))
    precision_total = tp_total / (tp_total + fp_total) if tp_total + fp_total > 0 else 0
    recall_total = tp_total / (tp_total + fn_total) if tp_total + fn_total > 0 else 0
    f1_total = 2 * precision_total * recall_total / (precision_total + recall_total) if precision_total + recall_total > 0 else 0
    print('**Overall: \n   Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(precision_total, recall_total, f1_total))
    if save_path is not None:
        with open(save_path, 'w') as f:
            for entity_type, results in final_results.items():
                f.write(f'##### {entity_type} #####\n')
                for k, (pred, score) in results.items():
                    f.write(f'{k}\t{pred}\t{score}\n')
        print(f'\nSaved final results to "{save_path}"')
