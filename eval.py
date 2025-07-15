import os
import re
import Prefixes
import pandas as pd
import math
import scipy.stats as st
import xml.etree.ElementTree as ET


# def load_openea_ref(loc):
#     df_ref = pd.read_csv(os.path.join(loc, 'ent_links'), sep='\t', header=None)
#     df_ref['dbp'] = df_ref[0].apply(lambda x: 'dbr:'+ unicode(x.replace('http://dbpedia.org/resource/', '')))
#     df_ref['wd'] = df_ref[1].apply(lambda x: x.replace('http://www.wikidata.org/entity/', 'wd:'))
#     y_gold = set(df_ref.apply(lambda x: (x['dbp'], x['wd']), axis=1))
#     return y_gold

def load_openea_ref(loc):
    gt_pairs = []
    with open(os.path.join(loc, 'ent_links'), 'r', encoding='UTF-8') as f:
        for line in f:
            head, tail = line.strip().split('\t')
            gt_pairs.append((head, tail))
    return gt_pairs

def openea_eval(maxAssignment, y_gold):
    y_pred = set()
    for e1 in maxAssignment:
        # if e1.startswith('dbr:'):
        if e1.startswith('http://dbpedia.org/resource/'):
            for e2 in maxAssignment[e1]:
                y_pred.add(tuple([e1, e2]))
                break # only select the first one
    
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
    

def load_ent_results(file_path, prefix, threshold=0.0):
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
    for e1, matches in sameASscore.items():
        if matches:
            max_score = max(matches.values())
            for e2 in matches:
                if matches[e2] == max_score:
                    if e1 not in match_e1_to_e2:
                        match_e1_to_e2[e1] = {}
                    match_e1_to_e2[e1][e2] = matches[e2]
                    if e2 not in match_e2_to_e1:
                        match_e2_to_e1[e2] = {}
                        match_e2_to_e1[e2][e1] = matches[e2]
                        continue

                    max_score_e2 = max(match_e2_to_e1[e2].values())
                    if matches[e2] > max_score_e2:
                        match_e2_to_e1[e2] = {e1: matches[e2]}
                    elif max_score_e2 == matches[e2]:
                        match_e2_to_e1[e2][e1] = matches[e2]
    res_max_assign = {} # bilateral max assignment
    for e2 in match_e2_to_e1:
        # exact match case, avoid duplicates
        if e2 in match_e2_to_e1[e2]:
            res_max_assign[e2] = {e2: match_e2_to_e1[e2][e2]}
            continue
        for e1 in match_e2_to_e1[e2]:
            if e1 in match_e1_to_e2 and e2 in match_e1_to_e2.get(e1, {}):
                if e2 not in res_max_assign:
                    res_max_assign[e2] = {}
                res_max_assign[e2][e1] = match_e1_to_e2[e1][e2]
                if e1 not in res_max_assign:
                    res_max_assign[e1] = {}
                res_max_assign[e1][e2] = match_e2_to_e1[e2][e1]
    return res_max_assign



def unicode(txt):
    # e.g., "Coamo,_Puerto_Rico" -> "Coamo_u002C_Puerto_Rico"
    encoded_str = re.sub(r'[^a-zA-Z0-9_]', lambda x: "_u{:04X}_".format(ord(x.group())), txt)
    return encoded_str

def simplify_uri(uri, code=True):
    for prefix, base in Prefixes.prefixes_dbp.items():
        if uri.startswith(base):
            if code:
                return prefix + ":" + unicode(uri[len(base):])
            else:
                return prefix + ":" + uri[len(base):]
    return uri

def load_dbp15k_ref(loc):
    id2ent1, id2ent2 = {}, {}
    for i in range(2):
        id2ent = {}
        with open(loc+'ent_ids_{}'.format(i+1), encoding='UTF-8') as f:
            for line in f.readlines():
                ids, ent = line.strip().split('\t')
                id2ent[int(ids)] = simplify_uri(ent)
        if i == 0:
            id2ent1 = id2ent.copy()
        else:
            id2ent2 = id2ent.copy()
    # load supervised data
    seed_pairs = {}
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
    # a simple version -> only gold standard are considered
    y_pred_inst = dict()
    y_pred_class = dict()
    y_pred_subproperty = dict()
    y_pred_sameAs = dict()
    with open(loc, 'r') as file:
        for line in file:
            terms = line.strip().split('\t')
            if len(terms) > 4:
                if terms[0] in inst_gt:
                    if terms[0] not in y_pred_inst:
                        y_pred_inst[terms[0]] = {}
                    y_pred_inst[terms[0]][terms[2]] = terms[4]
                elif terms[0] in cls_gt:
                    if terms[0] not in y_pred_class:
                        y_pred_class[terms[0]] = {}
                    y_pred_class[terms[0]][terms[2]] = terms[4]
                elif terms[0].startswith('<'+prefix+'property/') or \
                        terms[2].startswith('<'+prefix+'property/'): # bilateral subrelations
                    if terms[0] in rel_gt or \
                            terms[2] in rel_gt:
                        if terms[1] == 'owl:sameAs' and terms[0] in rel_gt:
                            if terms[0] not in y_pred_sameAs:
                                y_pred_sameAs[terms[0]] = {}
                            y_pred_sameAs[terms[0]][terms[2]] = terms[4]
                            continue
                        if terms[0] not in y_pred_subproperty:
                            y_pred_subproperty[terms[0]] = {}
                        y_pred_subproperty[terms[0]][terms[2]] = terms[4]
    # similar relations
    y_pred_similar = dict()
    for k in y_pred_subproperty:
        for v in y_pred_subproperty[k]:
            if k not in y_pred_similar:
                y_pred_similar[k] = {}
            y_pred_similar[k][v] = max(float(y_pred_subproperty[k][v]), float(y_pred_subproperty.get(v, {}).get(k, 0)))
            if v not in y_pred_similar:
                y_pred_similar[v] = {}
            y_pred_similar[v][k] = max(float(y_pred_subproperty[k][v]), float(y_pred_subproperty.get(v, {}).get(k, 0)))
    return y_pred_inst, y_pred_class, y_pred_similar, y_pred_sameAs



def oaei_kg_eval(cls_gt, inst_gt, rel_gt, 
                 cls_pred, inst_pred, rel_pred_same, rel_pred_similar,
                 prefix1, prefix2, threshold=0.1):
    tp_total, fp_total, fn_total = 0, 0, 0
    # Instances
    tp, fp, fn = 0, 0, 0
    for k, v in inst_gt.items():
        if k not in inst_pred:
            fn += 1
            continue
        # the one with maximum score
        sort_pred = sorted(inst_pred[k].items(), key=lambda x: x[1], reverse=True)
        candidate = [x for x in sort_pred if x[0].startswith('<' + prefix2 + 'resource/')]
        if candidate:
            pred, score = candidate[0][0], candidate[0][1]
            for ent in candidate:
                if ent[1] < score:
                    break
                # Multiple candidates, select the exact match one if exists
                if ent[0].split('/')[-1] == k.split('/')[-1]:
                    pred = ent[0]
                    score = ent[1]
                    break
        if float(score) <= threshold * 5:
            fn += 1
            continue
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
    print('Instances: \n Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(precision, recall, f1))

    # Classes
    tp, fp, fn = 0, 0, 0
    for k, v in cls_gt.items():
        if k not in cls_pred:
            fn += 1
            continue
        # the one with maximum score
        pred, score = sorted(cls_pred[k].items(), key=lambda x: x[1], reverse=True)[0]
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
    print('Classes: \n Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(precision, recall, f1))

    # Properties
    # First find sameAs matches
    y_pred_property_post = {}
    for k, preds in rel_pred_same.items():
        tmp = sorted(preds.items(), key=lambda x: x[1], reverse=True)
        max_score = tmp[0][1]
        candidate = [x for x in tmp if x[1] == max_score]
        for uri, score in candidate:
            if float(score) > threshold and uri.startswith('<'+prefix2+'property/'):
                y_pred_property_post.setdefault(k, {})[uri] = float(score)
                break
    # Find more similar properties from subrelations
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

    # evaluate
    tp, fp, fn = 0, 0, 0
    for k, v in rel_gt.items():
        if k not in y_pred_property_post:
            fn += 1
            continue
        # the one with maximum score: choose only the maximally assigned one
        pred, score = sorted(y_pred_property_post[k].items(), key=lambda x: x[1], reverse=True)[0]
        if score <= threshold:
            fn += 1
            continue
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
    print('Properties: \n Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(precision, recall, f1))
    precision_total = tp_total / (tp_total + fp_total) if tp_total + fp_total > 0 else 0
    recall_total = tp_total / (tp_total + fn_total) if tp_total + fn_total > 0 else 0
    f1_total = 2 * precision_total * recall_total / (precision_total + recall_total) if precision_total + recall_total > 0 else 0
    print('Overall: \n Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(precision_total, recall_total, f1_total))