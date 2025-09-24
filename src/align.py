from itertools import combinations
from collections import Counter
import multiprocessing as mp
from scipy import stats
import numpy as np
import pandas as pd
import utils



# Constants for accessing the components of a triple
SUBJ=0
PRED=1
OBJ=2


#################################################################
#               Predicates and Functionalities                  #
#################################################################
def initializePredicateSubsumption(predicates1, predicates2, pred2superPred12={}, pred2superPred21={}, relinit=0.1):
    """ 
    Initializes all identical relations to 1.0, all others as given or else to RELINC 

    Parameters
    ----------
    predicates1 : set
        set of predicates in KB1
    predicates2 : set
        set of predicates in KB2
    pred2superPred12 : dict, optional
        subsumption scores from predicates in KB1 to predicates in KB2
    pred2superPred21 : dict, optional
        subsumption scores from predicates in KB2 to predicates in KB1
    relinit : float, optional
        initial score for non-identical relations, by default 0.1
    
    Returns
    -------
    result : dict
        Nested dictionary of pairwise subsumption scores across KGs in both directions
    """
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
    """ 
    Updates the predicate subsumptions from two directions: kb1->kb2, kb2->kb1 

    Parameters
    ----------
    pred2superPred12 : dict
        current subsumption scores from predicates in KB1 to predicates in KB2
    pred2superPred21 : dict
        current subsumption scores from predicates in KB2 to predicates in KB1
    previousPredicate2superPredicate : dict
        previous subsumption scores to be updated
    """
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
    """ 
    Returns the functionalities of the predicates in the KB 

    Parameters
    ----------
    kb : Graph
        The input knowledge base
    gram : list
        List of integers indicating the n-grams to consider for functionality computation
    
    Returns
    -------
    dict
        A dictionary mapping predicates to their functionalities
    """
    predicate2numFacts={}
    predicate2subjects={}
    for subject in kb.subjects():
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


def computeFunctionalitiesForPredicates(kb, predicates):
    """ 
    Returns the functionalities of the given predicates list in the KB 

    Parameters
    ----------
    kb : Graph
        The input knowledge base
    predicates : list
        Relation list to compute functionalities for
    
    Returns
    -------
    float
        The functionality of the given relation list in the KB
    """
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
    """ 
    Updates mapping[key1][key2] so that the rule body=>mapping[key1][key2] holds 
        using the minimum operator (Godel logic), as shown in equation (1) in the paper.
    
    Parameters
    ----------
    mapping : dict
        Nested dictionary to be updated with the entity alignment scores
    key1 : hashable
        The entity from KB1
    key2 : hashable
        The entity from KB2
    body : list of float
        The values in the body of the rule
    """
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
    """ 
    Updates mapping[key1][key2] so that the rule body=>mapping[key1][key2] holds, but adds the values. 
    It is used for subrelation rules, as shown in equation (2) in the paper.

    Parameters
    ----------
    mapping : dict
        Nested dictionary to be updated with the subrelation scores
    key1 : hashable
        The predicate from KB1
    key2 : hashable
        The predicate from KB2
    factor : float
        Normalization factor (already multipled by benefit of the doubt paramter)
    body : list of float
        The values in the body of the rule
    """
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
    """ 
    Parameters
    ----------
    mapping : dict
        Dictionary to be updated with the maximum aligned scoring fact for each predicate
    pred : hashable
        The predicate from KB2
    fact : tuple
        The fact from KB2
    body : list of float
        The values in the body of the rule
    """
    # subrelation rules
    score = min(body)
    if pred not in mapping:
        mapping[pred] = (fact, score)
    else:
        if score > mapping[pred][1]:
            mapping[pred] = (fact, score)
    return


#################################################################
#                      Procedure                                #
#################################################################
def _1st_iteration(kb_src, kb_dst, pred2superPred, functionalities,
        queue, ent_match_tuple_queue, ent_max_assign):
    """ 
    The first iteration used for bootstrapping the algorithm using the initial literal alignments.
    
    Parameters
    ----------
    kb_src : Graph
        The source knowledge base
    kb_dst : Graph
        The target knowledge base
    pred2superPred : dict
        Nested dictionary of pairwise subsumption scores across KGs in both directions
    functionalities : dict
        A dictionary mapping predicates to their functionalities
    queue : mp.Queue
        A multiprocessing queue containing the entities to be aligned
    ent_match_tuple_queue : mp.Queue
        A multiprocessing queue to store the resulting entity alignment scores
    ent_max_assign : dict
        The bilateral max assignment computed from the initial literal alignments
    """
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
    """ 
    Bootstrapping the algorithm using the initial literal alignments.

    Parameters
    ----------
    kb_src : Graph
        The source knowledge base
    kb_dst : Graph
        The target knowledge base
    sameAsScore : dict
        Nested dictionary of entity alignment scores (includes initial literal alignments)
    pred2superPred : dict
        Nested dictionary of pairwise subsumption scores across KGs in both directions
    functionalities : dict
        A dictionary mapping predicates to their functionalities
    """
    ent_max_assign = bilateral_max_assign(sameAsScore)
    mgr_ = mp.Manager()
    subjs_kb1 = kb_src.subjects()
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
    """ 
    Maps subrelations (both directions) using the current entity alignments.

    Parameters
    ----------
    alpha : float
        Benefit of the doubt parameter for subrelation mapping
    kb_src : Graph
        The source knowledge base
    kb_dst : Graph
        The target knowledge base
    ent_maxAssign : dict
        The bilateral max assignment computed from the current entity alignments
    previouspredicate2superPredicate : dict
        Previous subsumption scores to be updated
    """
    # Match predicates
    pred2superPred1 = {}
    # Direction: kb1 -> kb2
    for fact1 in kb_src:
        if fact1[SUBJ] not in ent_maxAssign:
            continue
        # For each fact1, find the best matching fact2 for each relation r2
        # So that for the given relation pair (r1, r2), there is one most matched fact2
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
    """ 
    Computes the quasi equivalence relations between the two KGs' predicates, 
    the quasi equivalence is represented as r\cong r' in paper.

    Parameters
    ----------
    kb_src : Graph
        The source knowledge base
    kb_dst : Graph
        The target knowledge base
    pred2superPred : dict
        Nested dictionary of pairwise subsumption scores across KGs in both directions
    
    Returns
    -------
    quasiEqrel_ : dict
        Nested dictionary of quasi equivalence relations
    """
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
    """ 
    Computes the bilateral max assignment from the similarity scores, refer to equation (3) in paper.

    Parameters
    ----------
    sameASscore : dict
        Nested dictionary of entity alignment scores
    
    Returns
    -------
    res_max_assign : dict
        The bilateral max assignment of entities
    """
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
def _match_entities_by_rules(kb_src, kb_dst, quasiEqvirel, queue, ent_match_tuple_queue, sameAsScore, functionalities, params):
    """ 
    Match entities in parallel using the rules, corresponding to equation (2) in the paper.
    The function consists of two parts: candidate search and entity alignment.

    Parameters
    ----------
    kb_src : Graph
        The source knowledge base
    kb_dst : Graph
        The target knowledge base
    quasiEqvirel : dict
        Nested dictionary of quasi equivalence relations
    queue : mp.Queue
        A multiprocessing queue containing the entities to be aligned
    ent_match_tuple_queue : mp.Queue
        A multiprocessing queue to store the resulting entity alignment scores
    sameAsScore : dict
        Nested dictionary of all entity alignment scores (includes initial literal alignments)
    functionalities : dict
        A dictionary mapping predicates to their functionalities
    params : dict
        A dictionary of parameters, including 'gramN' (the maximum n-gram size to consider)
    """
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
        # select the entities with the most evidences
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
        

        # Alignment Algorithm
        # Apply rules in order to update the scores
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