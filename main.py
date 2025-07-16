import utils
import Prefixes
import Announce
import time
import init
from itertools import combinations
from collections import Counter
import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy import stats
import argparse
import logging
import os

# Constants for accessing the components of a triple
SUBJ=0
PRED=1
OBJ=2

#################################################################
#                    Loading data                               #
#################################################################


def get_params():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--kb1', type=str, default='kb1.ttl')
    # parser.add_argument('--kb2', type=str, default='kb2.ttl')
    parser.add_argument('--dataset', type=str, default='data/OpenEA/D_W_15K_V2/')
    parser.add_argument('--save_dir', type=str, default='save')
    parser.add_argument('--save_file', type=str, default='results.ttl')
    parser.add_argument('--trainingdata', type=str, default=None)
    parser.add_argument('--alpha', type=float, default=3.0)
    parser.add_argument('--init', type=float, default=0.7)
    parser.add_argument('--gramN', type=int, default=100)
    parser.add_argument('--epsilon', type=float, default=0.1)
    args, _ = parser.parse_known_args()
    params_ = vars(args)
    return params_


Announce.doing("Running FLORA...")

params = get_params()
Announce.set_logger(params)

# Load knowledge bases
Announce.doing("Loading Knowledge Bases")
if 'OpenEA' in params['dataset']:
    kb1, kb2, _ = utils.load_openea(params['dataset'], attr=True)
elif 'DBP15k' in params['dataset']:
    kb1, kb2 = utils.load_dbp15k(params['dataset'], attr=True, name=True)
elif 'OAEI' in params['dataset']:
    kb1, kb2 = utils.load_oaei(params['dataset'], format='ttl')
elif 'small-test' in params['dataset']:
    kb1 = utils.graphFromTurtleFile(os.path.join(params['dataset'], params['dataset'].split('/')[-2]+'1.ttl'))
    kb2 = utils.graphFromTurtleFile(os.path.join(params['dataset'], params['dataset'].split('/')[-2]+'2.ttl'))
else: 
    raise ValueError("Unknown dataset: %s" % params['dataset'])
Announce.done()

# Load training data (if any)
sameAsScores={}
# if len(sys.argv)>4:
if params['trainingdata'] is not None:
    print("Loading training data from %s" % params['trainingdata'])
    Announce.doing("Loading training data")
    with open(params["trainingdata"], "rt", encoding="utf-8") as trainingDataFile:
        for line in trainingDataFile:
            split=line.strip().split("\t")
            if split[0] not in sameAsScores:
                sameAsScores[split[0]]={}
            sameAsScores[split[0]][split[1]]=1.0 if len(split)<3 else float(split[2])
    Announce.done()


#################################################################
#               Predicates and Functionalities                  #
#################################################################


def initializePredicateSubsumption(predicates1, predicates2, pred2superPred12={}, pred2superPred21={}, relinit=0.1):
    """ Initializes all identical relations to 1.0, all others as given or else to RELINC """
    result = {}
    for pred1 in predicates1:
        if pred1 not in result:
            result[pred1] = {}
        for pred2 in predicates2:
            if pred2 not in result:
                result[pred2] = {}
            if pred1 == pred2:
                result[pred1][pred2] = 1.0
            else:
                score1 = max(pred2superPred12.get(pred1,{}).get(pred2,relinit),
                             pred2superPred21.get(pred1,{}).get(pred2,relinit))
                score2 = max(pred2superPred21.get(pred2,{}).get(pred1,relinit),
                             pred2superPred12.get(pred2,{}).get(pred1,relinit))
                result[pred1][pred2] = score1
                result[pred2][pred1] = score2
    return result


def updatePredicateSubsumption(pred2superPred12, pred2superPred21, previousPredicate2superPredicate):
    for pred1 in pred2superPred12:
        if previousPredicate2superPredicate.get(pred1) is None:
            previousPredicate2superPredicate[pred1] = {}
        for pred2 in pred2superPred12[pred1]:
            # Make relation subsumption monotonic
            previousPredicate2superPredicate[pred1][pred2] = max(previousPredicate2superPredicate[pred1].get(pred2, 0),
                                                                 pred2superPred12[pred1][pred2])
    for pred2 in pred2superPred21:
        if previousPredicate2superPredicate.get(pred2) is None:
            previousPredicate2superPredicate[pred2] = {}
        for pred1 in pred2superPred21[pred2]:
            # Make relation subsumption monotonic
            previousPredicate2superPredicate[pred2][pred1] = max(previousPredicate2superPredicate[pred2].get(pred1, 0),
                                                                 pred2superPred21[pred2][pred1])


def computeFunctionalities(kb, gram=[]):
    """ Returns the functionalities of the predicates in the KB """
    predicate2numFacts={}
    predicate2subjects={}
    for subject in kb.subjects():
        # print(subject)
        facts = list(kb.triplesWithSubject(subject))
        for n in gram:
            if n == 1:
                for fact in facts:
                    predicate_ = fact[PRED]
                    if predicate_ not in predicate2numFacts:
                        predicate2numFacts[predicate_]=0
                        predicate2subjects[predicate_]=set()
                    predicate2numFacts[predicate_]+=1
                    predicate2subjects[predicate_].add(fact[SUBJ])
                continue
            # gram > 1
            cnt = 0
            for evs in combinations(facts, n):
                cnt += 1
                predicate_ = tuple(sorted([utils.invert(fact[PRED]) for fact in evs]))
                subjs_ = tuple(sorted([fact[OBJ] for fact in evs]))
                if predicate_ not in predicate2numFacts:
                    predicate2numFacts[predicate_]=0
                    predicate2subjects[predicate_]=set()
                predicate2numFacts[predicate_]+=1
                predicate2subjects[predicate_].add(subjs_)
                if cnt > 100000: # avoid memory overflow
                    break
    return { predicate : len(predicate2subjects[predicate])/predicate2numFacts[predicate] for predicate in predicate2numFacts }    


def computeFunctionalitiesForPredicates_old(kb, predicates=[]):
    """ Returns the functionalities of the predicates in the KB """
    predicate2numFacts={}
    predicate2subjects={}
    predicates_inv = [utils.invert(pred) for pred in predicates]
    predsorted = tuple(sorted(predicates))
    for subject in kb.subjects():
        predsOfsubject = set(kb.index[subject].keys())
        if len(set(predicates_inv).intersection(predsOfsubject)) != len(set(predicates_inv)):
            continue
        facts = list(kb.triplesWithSubject(subject, set(predicates_inv)))
        for evs in combinations(facts, len(predicates_inv)):
            candPredicate = tuple(sorted(predicates_inv))
            predicate_ = tuple(sorted([fact[PRED] for fact in evs]))
            if predicate_ != candPredicate:
                continue
            subjs_ = tuple(sorted([fact[OBJ] for fact in evs]))
            if predsorted not in predicate2numFacts:
                predicate2numFacts[predsorted]=0
                predicate2subjects[predsorted]=set()
            predicate2numFacts[predsorted]+=1
            predicate2subjects[predsorted].add(subjs_)
    return len(predicate2subjects[predsorted])/predicate2numFacts[predsorted] if predicate2numFacts else 0



def computeFunctionalitiesForPredicates(kb, predicates):
    pred_numFacts = 0
    pred_subjects = set()
    counter = Counter(predicates)
    predicates_inv = sorted([utils.invert(pred) for pred in predicates])
    subKB = kb.headTriplesWithPredicateList({utils.invert(pred):counter[pred] for pred in set(predicates)})
    for obj in subKB:
        for evs in combinations(subKB[obj], len(predicates_inv)):
            _, predicate_, subjs_ = zip(*evs)
            if tuple(sorted(predicate_)) == tuple(predicates_inv):
                pred_numFacts += 1
                pred_subjects.add(tuple(sorted(subjs_)))
    return len(pred_subjects) / pred_numFacts if pred_numFacts > 0 else 0



#################################################################
#                 Implication Functions                         #
#################################################################

def updateScoreMin(mapping, key1, key2, *body):
    """ Updates mapping[key1][key2] so that the rule body=>mapping[key1][key2] holds """
    curScore = 0
    if key1 in mapping and key2 in mapping.get(key1, {}):
        curScore = mapping[key1][key2]
    tmp = max(curScore, min(min(body),1.0))

    if tmp > 0:
        if key1 not in mapping:
            mapping[key1]={}
        mapping[key1][key2] = tmp
    return


def updateScoreAdditiveMin(mapping, key1, key2, factor, *body):
    """ Updates mapping[key1][key2] so that the rule body=>mapping[key1][key2] holds, but adds the values """
    curScore = 0
    if key1 in mapping and key2 in mapping.get(key1, {}):
        curScore = mapping[key1][key2]
    value = curScore + min(body) * factor
    if value > 0:
        if key1 not in mapping:
            mapping[key1]={}
        mapping[key1][key2] = max(min(value, 1.0), 0)
    return


def updateMaxScoreMin(mapping, pred, fact, *body):
    # subrelation rules
    score = min(body)
    if pred not in mapping:
        mapping[pred] = (fact, score)
    else:
        if score > mapping[pred][1]:
            mapping[pred] = (fact, score)
    return


# Other useful functions
def filter_unique_patterns(list1, list2):
    '''
    Select list1 based on list2 pattern
    example: list1 = ['a', 'b', 'c']
             list2 = ['A', 'B', 'A']
             return: [(['a', 'b'], ['A', 'B']), (['b', 'c'], ['B', 'A'])]
    '''
    unique_patterns = set(list2)
    pattern_len = len(unique_patterns)
    results = []
    for indices in combinations(range(len(list2)), pattern_len):
        selected_elm2 = [list2[i] for i in indices]
        if len(set(selected_elm2)) == pattern_len:
            selected_elm1 = [list1[i] for i in indices]
            results.append((selected_elm1, selected_elm2))
    return results


def violated(x, y):
    """
    Check if x and y violate the rule, 
        i.e., if there are two adjacent elements in x that are the same, 
              the corresponding elements in y should not be the same.
    """
    for i in range(len(x) - 1):
        if x[i] == x[i + 1]:
            if y[i] == y[i + 1]:
                return True
    return False


#################################################################
#                      Procedure                                #
#################################################################
def _1st_iteration(kb_src, kb_dst, pred2superPred, functionalities,
        queue, ent_match_tuple_queue, ent_max_assign):
    ent_match_scores = dict()
    while not queue.empty():
        try:
            subj_kb1 = queue.get_nowait()
        except Exception:
            break

        for fact1 in kb_src.triplesWithSubject(subj_kb1):
            # We don't match literals
            if utils.isLiteral(fact1[OBJ]):
                continue
            # Continue if the subject has not been matched
            if fact1[SUBJ] not in ent_max_assign:
                continue
            for subj_kb2 in ent_max_assign[fact1[SUBJ]]:
                if subj_kb2 not in kb_dst.index:
                    continue
                for fact2 in kb_dst.triplesWithSubject(subj_kb2, pred2superPred[fact1[PRED]]):
                    # We don't match literals
                    if utils.isLiteral(fact2[OBJ]):
                        continue
                    # Update
                    updateScoreMin(
                        # Objects are the same, ...
                        ent_match_scores, fact1[OBJ], fact2[OBJ],
                        # ... if the subjects are the same, ...
                        ent_max_assign[fact1[SUBJ]][fact2[SUBJ]],
                        # ... and the predicate is locally functional, ...
                        kb_src.localFunctionality(fact1[SUBJ], fact1[PRED]), 
                        kb_dst.localFunctionality(fact2[SUBJ], fact2[PRED]),
                        # ... and the predicate is globally functional,
                        functionalities[fact1[PRED]], functionalities[fact2[PRED]],
                        # ... and the target predicate is subsumed.
                        max(pred2superPred[fact1[PRED]][fact2[PRED]],
                            pred2superPred[fact2[PRED]][fact1[PRED]])
                    )
    # Update the queue
    ent_match_tuple_queue.put(ent_match_scores)
    exit(1)


def bootstrap_algo(kb_src, kb_dst, sameAsScore, pred2superPred, functionalities):
    ent_max_assign = bilateral_max_assign(sameAsScore)
    mgr_ = mp.Manager()
    subjs_kb1 = kb1.subjects()
    ent_queue_ = mgr_.Queue(len(subjs_kb1))
    for subj_kb1 in subjs_kb1:
        ent_queue_.put(subj_kb1)
    tasks = []
    num_workers = 90
    ent_match_tuple_queue_ = mgr_.Queue()
    for _ in range(num_workers):
        task = mp.Process(
            target=_1st_iteration,
            args=(
                  kb_src, kb_dst,
                  pred2superPred,
                  functionalities,
                  ent_queue_,
                  ent_match_tuple_queue_,
                  ent_max_assign,
                ))
        task.start()
        tasks.append(task)
    for task in tasks:
        task.join()
    print("Bootstrap entities done.")
    while not ent_match_tuple_queue_.empty():
        ent_match_score_dict = ent_match_tuple_queue_.get()
        # update sameAsScores using max aggregation
        for subj1 in ent_match_score_dict:
            if subj1 not in sameAsScore:
                    sameAsScore[subj1] = {}
            for subj2 in ent_match_score_dict[subj1]:
                if ent_match_score_dict[subj1][subj2] > sameAsScore[subj1].get(subj2, 0):
                    sameAsScore[subj1][subj2] = ent_match_score_dict[subj1][subj2]


def map_subrelations(alpha, kb_src, kb_dst, ent_maxAssign, previouspredicate2superPredicate):
    # Match predicates
    pred2superPred1 = {}
    # print("\n Mapping predicates forward...")
    # Direction: kb1 -> kb2
    for fact1 in kb_src:
        if fact1[SUBJ] not in ent_maxAssign:
            continue
        rel2maxFact = {} # {rel2: (fact2, score)}
        for subject2 in ent_maxAssign[fact1[SUBJ]]:
            if subject2 not in kb_dst.index:
                continue
            for fact2 in kb_dst.triplesWithSubject(subject2):
                # change
                if fact1[OBJ] not in ent_maxAssign:
                    continue
                if fact2[OBJ] not in ent_maxAssign.get(fact1[OBJ], {}):
                    continue
                scoreObject = ent_maxAssign[fact1[OBJ]][fact2[OBJ]]
                updateMaxScoreMin(rel2maxFact, fact2[PRED], fact2, ent_maxAssign[fact1[SUBJ]][fact2[SUBJ]], scoreObject)
        # update
        for pred2, (fact2, score) in rel2maxFact.items():
            updateScoreAdditiveMin(pred2superPred1, fact1[PRED], fact2[PRED], alpha/kb_src.numFactsWithPredicate(fact1[PRED]), score)
    
    # Direction: kb2 -> kb1
    pred2superPred2 = {}
    # print(" Mapping predicates backward...")
    for fact2 in kb_dst:
        if fact2[SUBJ] not in ent_maxAssign:
            continue
        rel1maxFact = {} # {rel1: (fact1, score)}
        for subject1 in ent_maxAssign[fact2[SUBJ]]:
            if subject1 not in kb_src.index:
                continue
            for fact1 in kb_src.triplesWithSubject(subject1):
                # change
                if fact2[OBJ] not in ent_maxAssign:
                    continue
                if fact1[OBJ] not in ent_maxAssign.get(fact2[OBJ], {}):
                    continue
                scoreObject = ent_maxAssign[fact2[OBJ]][fact1[OBJ]]
                updateMaxScoreMin(rel1maxFact, fact1[PRED], fact1, ent_maxAssign[fact2[SUBJ]][fact1[SUBJ]], scoreObject)
        # update
        for pred1, (fact1, score) in rel1maxFact.items():
            updateScoreAdditiveMin(pred2superPred2, fact2[PRED], fact1[PRED], alpha/kb_dst.numFactsWithPredicate(fact2[PRED]), score)
    # complete the subrelation mapping
    updatePredicateSubsumption(pred2superPred1, pred2superPred2, previouspredicate2superPredicate)


def computeQuasiEqrel(kb_src, kb_dst, pred2superPred):
    quasiEqrel_ = {} # from kb1 to kb2
    for pred1 in pred2superPred:
        for pred2 in pred2superPred[pred1]:
            value = max(pred2superPred[pred1][pred2], 
                        pred2superPred.get(pred2, {}).get(pred1, 0))
            if pred1 in kb_src.predicates():
                if pred2 in kb_dst.predicates():
                    if pred1 not in quasiEqrel_:
                        quasiEqrel_[pred1] = {}
                    quasiEqrel_[pred1][pred2] = value
            elif pred2 in kb_src.predicates() and pred2 not in quasiEqrel_:
                quasiEqrel_[pred2] = {}
                quasiEqrel_[pred2][pred1] = value
    return quasiEqrel_


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



# Matching in parallel
def _match_entities_by_rules(kb_src, kb_dst, quasiEqvirel, queue, ent_match_tuple_queue, sameAsScore, functionalities):
    """ Match entities in parallel using the rules """
    ent_match_scores = dict()
    ent_max_assign = bilateral_max_assign(sameAsScore)
    while not queue.empty():
        try:
            subj_kb1 = queue.get_nowait()
        except Exception:
            break
        
        # We don't need to match literals
        if utils.isLiteral(subj_kb1):
            continue

        # Skip if the entity is already matched
        if subj_kb1 in ent_max_assign and \
            round(max(ent_max_assign.get(subj_kb1, {None: 0}).values()), 1) >= 1.0:
            continue
        
        # Search Algorithm
        kb1_facts_ordered = []
        for fact1 in kb_src.triplesWithSubject(subj_kb1):
            if max(ent_max_assign.get(fact1[OBJ], {None: 0}).values()) <= 0:
                continue
            if max(quasiEqvirel.get(fact1[PRED], {None:0}).values()) <= 0:
                    continue
            kb1_facts_ordered.append((fact1[OBJ], utils.invert(fact1[PRED]), subj_kb1))
        # seach order: the most informative facts first
        kb1_facts_ordered.sort(reverse=True, key=lambda x: min(max(ent_max_assign[x[SUBJ]].values()), 
                                                               max(quasiEqvirel[x[PRED]].values())))
        subj2evi1 = dict() # a dict of list of ordered evidences
        subj2evi2 = dict() # {subj2:[ev2, ...]}
        for fact_kb1 in kb1_facts_ordered[:params['gramN']]:
            pred_kb1, obj_kb1 = fact_kb1[PRED], fact_kb1[SUBJ]
            # find the corresponding facts in kb2
            tmp_subj2_evi2 = dict()
            subj2_maxsubrel_score = dict()
            for obj_kb2 in ent_max_assign[obj_kb1]:
                if obj_kb2 not in kb_dst.index:
                    continue
                aligned_evi2 = []
                maxsubrel_score = 0
                for evi2_ in kb_dst.triplesWithSubject(obj_kb2):
                    if utils.isLiteral(evi2_[OBJ]):
                        continue
                    subrel_score = quasiEqvirel[pred_kb1].get(evi2_[PRED], 0)
                    if subrel_score <= 0:
                        continue
                    if subrel_score > maxsubrel_score:
                        maxsubrel_score = subrel_score
                        aligned_evi2 = [evi2_]
                    if subrel_score == maxsubrel_score:
                        aligned_evi2.append(evi2_)
                if len(aligned_evi2) == 0:
                    continue
                for evi2 in aligned_evi2:
                    subj2_ = evi2[OBJ]
                    if subj2_ not in subj2_maxsubrel_score:
                        subj2_maxsubrel_score[subj2_] = 0
                    # certain evi1 (subj1, obj1, pred1),
                    # one subj2 has just one correpsonding evidence2 at most
                    if quasiEqvirel[pred_kb1][evi2[PRED]] > subj2_maxsubrel_score[subj2_]:
                        subj2_maxsubrel_score[subj2_] = quasiEqvirel[pred_kb1][evi2[PRED]]
                        tmp_subj2_evi2[subj2_] = evi2
            # update subj2evi1 for exact evidence1 == fact_kb1
            for subj2, single_evi2 in tmp_subj2_evi2.items():
                if subj2 not in subj2evi1:
                    subj2evi1[subj2] = [fact_kb1]
                    subj2evi2[subj2] = [single_evi2]
                    continue
                # Reduce duplicates
                if single_evi2 in subj2evi2[subj2]:
                    index_evi2 = subj2evi2[subj2].index(single_evi2)
                    # campare scores
                    score1 = min(ent_max_assign[subj2evi1[subj2][index_evi2][SUBJ]][single_evi2[SUBJ]],
                                    quasiEqvirel[subj2evi1[subj2][index_evi2][PRED]][single_evi2[PRED]])
                    score2 = min(ent_max_assign[fact_kb1[SUBJ]][single_evi2[SUBJ]],
                                    quasiEqvirel[fact_kb1[PRED]][single_evi2[PRED]])
                    if score2 > score1:
                        subj2evi1[subj2][index_evi2] = fact_kb1
                        subj2evi2[subj2][index_evi2] = single_evi2
                    continue
                subj2evi1[subj2].append(fact_kb1)
                subj2evi2[subj2].append(single_evi2)
        
        # Selection Algorithm
        subj2_count = dict()
        maxCount = 0
        for subj2 in subj2evi2:
            if subj2 in ent_max_assign and \
                round(max(ent_max_assign.get(subj2, {None: 0}).values()), 1) >= 1.0:
                continue
            cur_count = len(set(subj2evi2[subj2]))
            if cur_count > maxCount:
                subj2_count = dict()
                maxCount = cur_count
                subj2_count[subj2] = len(set(subj2evi2[subj2]))
            elif cur_count == maxCount:
                subj2_count[subj2] = len(set(subj2evi2[subj2]))
        if len(subj2_count) == 0:
            continue
        

        # Apply rules
        gramN = min(20, maxCount)
        for subj_kb2 in subj2_count:
            assert len(subj2evi1[subj_kb2]) == len(subj2evi2[subj_kb2])

            # Re-order the list
            index_sorted = sorted(range(len(subj2evi1[subj_kb2])), reverse=True,
                                key=lambda i: min(ent_max_assign[subj2evi1[subj_kb2][i][SUBJ]][subj2evi2[subj_kb2][i][SUBJ]],
                                                    quasiEqvirel[subj2evi1[subj_kb2][i][PRED]][subj2evi2[subj_kb2][i][PRED]]))
            subj2evi1[subj_kb2] = [subj2evi1[subj_kb2][i] for i in index_sorted]
            subj2evi2[subj_kb2] = [subj2evi2[subj_kb2][i] for i in index_sorted]

            # find the common patterns
            visited_facts = set()
            ev1s, ev2s = subj2evi1[subj_kb2], subj2evi2[subj_kb2]
            # Try all possible sets
            for n in range(1, gramN+1):
                ev1, ev2 = ev1s[:n], ev2s[:n]
                if (tuple(ev1), tuple(ev2)) in visited_facts:
                        continue
                visited_facts.add((tuple(ev1), tuple(ev2)))
                obj1_combo, pred1_combo, subj1_combo = zip(*ev1)
                obj2_combo, pred2_combo, subj2_combo = zip(*ev2)
                # check if subjects itself are the same
                assert len(set(subj1_combo)) == 1
                assert len(set(subj2_combo)) == 1
                # check same pattern
                if not (pd.factorize(np.array(obj1_combo))[0]
                        == pd.factorize(np.array(obj2_combo))[0]).all():
                    continue
                localfunc1 = kb_src.localFunctionality(obj1_combo, pred1_combo)
                localfunc2 = kb_dst.localFunctionality(obj2_combo, pred2_combo)
                pred1_sort = tuple(sorted(list(pred1_combo)))
                pred2_sort = tuple(sorted(list(pred2_combo)))
                globalfunc1 = functionalities.get(pred1_sort, 1.0)
                globalfunc2 = functionalities.get(pred2_sort, 1.0)
                
                obj_eq = stats.hmean([ent_max_assign[obj1_combo[i]][obj2_combo[i]] for i in range(len(obj1_combo))])
                pred_eq = stats.hmean([quasiEqvirel[pred1_combo[i]][pred2_combo[i]] for i in range(len(pred1_combo))])
                # update
                if n == 1:
                    updateScoreMin(
                        ent_match_scores, subj1_combo[0], subj2_combo[0],
                        obj_eq, pred_eq, localfunc1, localfunc2,
                        functionalities[pred1_combo[0]], functionalities[pred2_combo[0]]
                    )
                else:
                    updateScoreMin(
                        ent_match_scores, subj1_combo[0], subj2_combo[0],
                        obj_eq, pred_eq, localfunc1, localfunc2,
                        globalfunc1, globalfunc2,
                    )
    # Update the queue
    ent_match_tuple_queue.put(ent_match_scores)
    exit(1)


#################################################################
#            Initialization + Bootstrapping                     #
#################################################################


Announce.doing("Initializing Subrelations")
predicates1 = kb1.predicates()
predicates2 = kb2.predicates()
predicate2superPredicate=initializePredicateSubsumption(predicates1, predicates2, relinit=0.1)
Announce.done()

Announce.doing("Computing functionalities")
functionalities1=computeFunctionalities(kb1, gram=[1, 2])
functionalities2=computeFunctionalities(kb2, gram=[1, 2])
functionalities = {}
for pred in functionalities1:
    functionalities[pred] = functionalities1[pred]
for pred in functionalities2:
    if pred not in functionalities:
        functionalities[pred] = functionalities2[pred]
        continue
    functionalities[pred] = min(functionalities[pred], functionalities2[pred])
Announce.done()

Announce.doing("Computing literal scores with threshold", params['init'])
path_emb = os.path.join('data/emb/', params['dataset'].split('/')[-2])
print("\n       path_emb1:", os.path.join(path_emb, 'kb1.pkl'))
print("       path_emb2:", os.path.join(path_emb, 'kb2.pkl'))
init.mapLiterals(kb1, kb2, path_emb, sameAsScores, params['init'])
Announce.done()


# Bootstrapping the entity alignment by literals
starttime = time.time()
Announce.doing("Bootstrapping")
BOD = params['alpha']
bootstrap_algo(kb1, kb2, sameAsScores, predicate2superPredicate, functionalities)
ent_maxAssign = bilateral_max_assign(sameAsScores)
# Subrelations
predicate2superPredicate = {}
map_subrelations(BOD, kb1, kb2, ent_maxAssign, predicate2superPredicate)
quasiEqvirel = computeQuasiEqrel(kb1, kb2, predicate2superPredicate)
Announce.done()
logging.info("Time used for bootstrapping: %s minutes"%((time.time() - starttime)/60))
logging.info("---------------Main Loop---------------")


#################################################################
#                         Main Loop                             #
#################################################################


MAXITERATIONS = 100
iterations=0
while True:
    Announce.doing("Iteration",iterations+1)
    
    sameAsSum=sum(val for dict_ in sameAsScores.values() for val in dict_.values())
    
    Announce.doing("Applying the Entity Alignment rules")
    starttime1 = time.time()
    mgr = mp.Manager()
    subjs_kb1 = kb1.subjects()
    ent_queue = mgr.Queue(len(subjs_kb1))
    for subj_kb1 in subjs_kb1:
        ent_queue.put(subj_kb1)
    
    tasks = []
    num_workers = 90
    ent_match_tuple_queue = mgr.Queue()
    for _ in range(num_workers):
        task = mp.Process(
            target=_match_entities_by_rules,
            args=(
                  kb1, kb2,
                  quasiEqvirel,
                  ent_queue,
                  ent_match_tuple_queue,
                  sameAsScores,
                  functionalities,
                )
            )
        task.start()
        tasks.append(task)

    for task in tasks:
        task.join()
    
    # Update the entity alignment scores
    while not ent_match_tuple_queue.empty():
        ent_match_score_dict = ent_match_tuple_queue.get()
        # update sameAsScores using max aggregation
        for subj1 in ent_match_score_dict:
            max_score1 = max(ent_maxAssign.get(subj1, {None: 0}).values())
            for subj2 in ent_match_score_dict[subj1]:
                max_score2 = max(ent_maxAssign.get(subj2, {None: 0}).values())
                # Avoid propagating the False Positives
                # not above the maximum assignment score
                if ent_match_score_dict[subj1][subj2] <= max(max_score1, max_score2):
                    continue
                # update
                if subj1 not in sameAsScores:
                    sameAsScores[subj1] = {}
                if subj2 not in sameAsScores[subj1]:
                    sameAsScores[subj1][subj2] = ent_match_score_dict[subj1][subj2]
                    continue
                # Take Max Score
                if ent_match_score_dict[subj1][subj2] > sameAsScores[subj1][subj2]:
                    sameAsScores[subj1][subj2] = ent_match_score_dict[subj1][subj2]
    Announce.done()
    logging.info("----Iteration %s----"%iterations)
    logging.info("Aligning entities: %s minutes"%((time.time() - starttime1)/60))

    Announce.doing("Recomputing predicate inclusions")
    starttime1 = time.time()
    ent_maxAssign = bilateral_max_assign(sameAsScores)
    map_subrelations(BOD, kb1, kb2, ent_maxAssign, predicate2superPredicate)
    quasiEqvirel = computeQuasiEqrel(kb1, kb2, predicate2superPredicate)
    Announce.done()
    logging.info("Aligning predicates: %s minutes"%((time.time() - starttime1)/60))

    # Check convergence
    Announce.doing("Checking convergence")
    newSameAsSum=sum(val for dict_ in sameAsScores.values() for val in dict_.values())   
    Announce.done(sameAsSum,newSameAsSum)
    logging.info("SameAs sum: %s -> %s"%(sameAsSum, newSameAsSum))
    
    Announce.done() # Iteration
    iterations+=1
    if iterations>MAXITERATIONS or abs(newSameAsSum - sameAsSum) < params['epsilon']:
        break


#################################################################
# Write out results
#################################################################
Announce.doing("Writing out results")
with open(os.path.join(params['save_dir'], params['save_file']), "wt", encoding="utf-8") as out:
    for p in Prefixes.prefixes:
        out.write("@prefix "+p+": <"+Prefixes.prefixes[p]+"> .\n")
    # Predicates
    kb1_predicates=kb1.predicates()
    kb2_predicates=kb2.predicates()
    predicates = kb1_predicates | kb2_predicates
    for predicate1 in predicates:
        if predicate1 in predicate2superPredicate:
            for predicate2 in predicate2superPredicate[predicate1]:
                if predicate2superPredicate[predicate1][predicate2] > 0.1:
                    out.write(predicate1+"\trdfs:subPropertyOf\t"+predicate2+"\t.#\t"+str(predicate2superPredicate[predicate1][predicate2])+"\n")
    # Literals and instances
    for entity1 in sameAsScores:
        for entity2 in sameAsScores[entity1]:
            if sameAsScores[entity1][entity2] > 0: # first report all possible scores
                out.write(entity1+"\towl:sameAs\t"+entity2+"\t.#\t"+str(sameAsScores[entity1][entity2])+"\n")
Announce.done()
Announce.done() # End of running FLORA
logging.info("Time used for the whole procedure: %s minutes"%((time.time() - starttime)/60))