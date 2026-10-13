"""
This file is part of FLORA, an unsupervised system for automatic knowledge graph (KG) alignment. 
The file is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0) by Yiwen Peng, Thomas Bonald, Fabian Suchanek and Lingyun Huang.

Description: Shared alignment utilities, predicate scoring functions, compact assignment indexes, and multiprocessing helpers for FLORA.
"""

from itertools import combinations
from collections import Counter
import os
import time
import logging
import shutil
from queue import Empty, Full

import log
import utils
import side_keys


# Constants for accessing the components of a triple
SUBJ=0
PRED=1
OBJ=2

ENTITY_TASK_CHUNK_SIZE = 100 # number of entities to feed workers in each chunk
SUBRELATION_FACT_CHUNK_SIZE = 5000 # number of subject-relation pairs to feed workers in each chunk
SUBRELATION_SUBJECT_OBJECT_CACHE_DEGREE_THRESHOLD = 512 # minimum target-subject degree to build a temporary object->predicate cache
SUBRELATION_SUBJECT_OBJECT_CACHE_MAX_SUBJECTS = 1024 # maximum cached target subjects per subrelation worker
MATCH_RESULT_FLUSH_PAIRS = 50000 # number of match results to accumulate before flushing to the output
RESULT_DRAIN_INTERVAL = 0.1 # time interval (in seconds) to drain results from the output queue
PROGRESS_LOG_INTERVAL = 300.0 # time interval (in seconds) to log progress information
PROGRESS_CHECK_FACT_INTERVAL = 100000 # number of facts to process before checking progress
MAX_ENT_SCORE_CACHE_SIZE = 262144 # maximum size of the entity max-score cache to avoid unbounded memory growth


#################################################################
#                       General Utilities                       #
#################################################################

def fast_hmean(values):
    """Harmonic mean for short positive score sequences."""
    count = 0
    reciprocal_sum = 0.0
    for value in values:
        if value <= 0:
            return 0.0
        reciprocal_sum += 1.0 / value
        count += 1
    return count / reciprocal_sum if count and reciprocal_sum else 0.0

#################################################################
#                 Multiprocessing Queue Helpers                 #
#################################################################

def iter_entity_chunks(graph, chunk_size=ENTITY_TASK_CHUNK_SIZE, include_literals=False, only_literals=False):
    """
    Yield graph entities in bounded chunks to avoid materializing the full task list in memory before workers start consuming it.

    Parameters
    ----------
    graph : Graph
        Graph whose entities should be streamed into chunks.
    chunk_size : int, optional
        Maximum number of entities per chunk.
    include_literals : bool, optional
        When False, filter out literal entities.
    only_literals : bool, optional
        When True, emit only literal entities. Bootstrap uses this to propagate only literals.
    """
    chunk = []
    subject_iter = graph.iter_view_subjects()

    for subject in subject_iter:
        if isinstance(subject, int) and hasattr(graph, 'is_literal_id'):
            is_literal = graph.is_literal_id(subject)
        else:
            subject_value = graph.entity_for_id(subject) if isinstance(subject, int) else subject
            is_literal = bool(utils.isLiteral(subject_value))
        if only_literals and not is_literal:
            continue
        if not include_literals and is_literal:
            continue
        chunk.append(subject)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def iter_source_entity_chunks(graph, source_entities, chunk_size=ENTITY_TASK_CHUNK_SIZE,
                              include_literals=False, only_literals=False):
    """
    Yield bounded chunks from an explicit source set.
    """
    chunk = []
    graph_side = side_keys.graph_side(graph, side_keys.KB1)
    if only_literals and hasattr(graph, 'iter_literal_object_ids'):
        source_lookup = source_entities if hasattr(source_entities, '__contains__') else set(source_entities)
        for entity_id in graph.iter_literal_object_ids():
            side_key = (graph_side, entity_id)
            if side_key in source_lookup:
                chunk.append(side_key)
            elif entity_id in source_lookup:
                chunk.append(entity_id)
            else:
                continue
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
        return

    for subject in source_entities:
        if side_keys.is_side_key(subject):
            if subject[0] != graph_side:
                continue
            if hasattr(graph, 'is_literal_id'):
                is_literal = graph.is_literal_id(subject[1])
                subject_value = None
            else:
                try:
                    subject_value = graph.entity_for_id(subject[1])
                except (IndexError, KeyError, TypeError, ValueError):
                    continue
        elif isinstance(subject, int) and hasattr(graph, 'entity_for_id'):
            if hasattr(graph, 'is_literal_id'):
                is_literal = graph.is_literal_id(subject)
                subject_value = None
            else:
                try:
                    subject_value = graph.entity_for_id(subject)
                except (IndexError, KeyError, TypeError, ValueError):
                    continue
        else:
            subject_value = subject
        if subject_value is not None:
            is_literal = bool(utils.isLiteral(subject_value))
        if only_literals and not is_literal:
            continue
        if not include_literals and is_literal:
            continue
        chunk.append(subject)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def drain_queue_items(queue_obj, handler):
    """
    Drain all currently available items from a multiprocessing queue.
    """
    drained = 0
    while True:
        try:
            item = queue_obj.get_nowait()
        except Empty:
            break
        handler(item)
        drained += 1
    return drained

def drain_side_queue_items(side_queues):
    """Drain optional side queues such as worker profile metadata."""
    drained_counts = {}
    for label, queue_obj, handler in side_queues or []:
        drained_counts[label] = drained_counts.get(label, 0) + drain_queue_items(queue_obj, handler)
    return drained_counts

def stop_workers(tasks):
    """Terminate any running workers and reap all child processes."""
    for task in tasks:
        if task.is_alive():
            task.terminate()
    for task in tasks:
        task.join()

def raise_if_worker_failed(tasks, stage_label=None):
    """Stop a worker stage and raise when any child exits unsuccessfully."""
    failed_workers = [
        (task.pid, task.exitcode)
        for task in tasks
        if task.exitcode not in (None, 0)
    ]
    if not failed_workers:
        return

    stop_workers(tasks)
    details = ", ".join(
        "pid=%s exitcode=%s" % worker
        for worker in failed_workers
    )
    raise RuntimeError(
        "%s worker failed | %s"
        % (stage_label or "Worker stage", details)
    )

def _put_task_with_drain(task_queue, item, result_queue=None, merge_fn=None,
                         wait_interval=RESULT_DRAIN_INTERVAL,
                         progress_logger=None, progress_fields=None,
                         side_queues=None, worker_tasks=None, stage_label=None):
    """
    Try to enqueue one task without blocking indefinitely. If the task queue is
    full, drain worker results before retrying so the producer cannot deadlock
    while workers are blocked on the result queue.
    """
    drained_results = 0
    wait_loops = 0
    while True:
        if worker_tasks is not None:
            raise_if_worker_failed(worker_tasks, stage_label)
        try:
            task_queue.put_nowait(item)
            return drained_results, wait_loops
        except Full: # if the task queue is full, drain results and retry
            wait_loops += 1
            if result_queue is not None and merge_fn is not None:
                drained_results += drain_queue_items(result_queue, merge_fn)
            side_drained_counts = drain_side_queue_items(side_queues)
            if worker_tasks is not None:
                raise_if_worker_failed(worker_tasks, stage_label)
            if progress_logger is not None:
                fields = dict(progress_fields or {})
                fields['queue_wait_loops'] = wait_loops
                fields['merged_result_chunks'] = fields.get('merged_result_chunks', 0) + drained_results
                for label, count in side_drained_counts.items():
                    fields[f"drained_{label}"] = count
                progress_logger(**fields)
            time.sleep(wait_interval)

def feed_entity_chunks(task_queue, graph, num_workers, chunk_size=ENTITY_TASK_CHUNK_SIZE,
                       result_queue=None, merge_fn=None, include_literals=False,
                       stage_label=None, progress_interval=PROGRESS_LOG_INTERVAL,
                       side_queues=None, only_literals=False,
                       source_entities=None, worker_tasks=None):
    """
    Feed entity chunks lazily so only a bounded number of tasks reside in the
    queue at a time. Optionally drain partial worker results while feeding.
    """
    progress_logger = log._build_progress_logger(stage_label, interval=progress_interval)
    fed_chunks = 0  # number of fed chunks
    fed_entities = 0 # number of fed entities
    sentinels_sent = 0 # number of ended workers
    merged_result_chunks = 0 # number of merged chunks

    if source_entities is None:
        chunk_iter = iter_entity_chunks(
            graph,
            chunk_size=chunk_size,
            include_literals=include_literals,
            only_literals=only_literals,
        )
    else:
        chunk_iter = iter_source_entity_chunks(
            graph,
            source_entities,
            chunk_size=chunk_size,
            include_literals=include_literals,
            only_literals=only_literals,
        )

    for entity_chunk in chunk_iter:
        drained_results, wait_loops = _put_task_with_drain(
            task_queue,
            entity_chunk,
            result_queue=result_queue,
            merge_fn=merge_fn,
            progress_logger=progress_logger,
            progress_fields={
                'fed_chunks': fed_chunks,
                'fed_entities': fed_entities,
                'sentinels_sent': sentinels_sent,
                'merged_result_chunks': merged_result_chunks,
            },
            side_queues=side_queues,
            worker_tasks=worker_tasks,
            stage_label=stage_label,
        )
        merged_result_chunks += drained_results
        fed_chunks += 1
        fed_entities += len(entity_chunk)
        if result_queue is not None and merge_fn is not None:
            merged_result_chunks += drain_queue_items(result_queue, merge_fn)
        side_drained_counts = drain_side_queue_items(side_queues)
        if progress_logger is not None:
            fields = {
                'fed_chunks': fed_chunks,
                'fed_entities': fed_entities,
                'sentinels_sent': sentinels_sent,
                'merged_result_chunks': merged_result_chunks,
                'queue_wait_loops': wait_loops,
            }
            for label, count in side_drained_counts.items():
                fields[f"drained_{label}"] = count
            progress_logger(**fields)
    for _ in range(num_workers):
        drained_results, wait_loops = _put_task_with_drain(
            task_queue,
            None,
            result_queue=result_queue,
            merge_fn=merge_fn,
            progress_logger=progress_logger,
            progress_fields={
                'fed_chunks': fed_chunks,
                'fed_entities': fed_entities,
                'sentinels_sent': sentinels_sent,
                'merged_result_chunks': merged_result_chunks,
            },
            side_queues=side_queues,
            worker_tasks=worker_tasks,
            stage_label=stage_label,
        )
        merged_result_chunks += drained_results
        sentinels_sent += 1
        if result_queue is not None and merge_fn is not None:
            merged_result_chunks += drain_queue_items(result_queue, merge_fn)
        side_drained_counts = drain_side_queue_items(side_queues)
        if progress_logger is not None:
            fields = {
                'fed_chunks': fed_chunks,
                'fed_entities': fed_entities,
                'sentinels_sent': sentinels_sent,
                'merged_result_chunks': merged_result_chunks,
                'queue_wait_loops': wait_loops,
            }
            for label, count in side_drained_counts.items():
                fields[f"drained_{label}"] = count
            progress_logger(**fields)

    return {
        'fed_chunks': fed_chunks,
        'fed_entities': fed_entities,
        'sentinels_sent': sentinels_sent,
        'merged_result_chunks': merged_result_chunks,
    }

def wait_for_workers_and_drain(tasks, result_queue, merge_fn, poll_interval=RESULT_DRAIN_INTERVAL,
                               stage_label=None, progress_interval=PROGRESS_LOG_INTERVAL,
                               side_queues=None):
    """
    Merge partial worker results while processes are still running so the
    result queue does not accumulate the full iteration output. Side queues are
    drained too, otherwise workers can block while publishing final metadata
    that the parent only reads after all workers exit.
    """
    progress_logger = log._build_progress_logger(stage_label, interval=progress_interval)
    merged_result_chunks = 0
    while True:
        alive_workers = sum(task.is_alive() for task in tasks)
        merged_result_chunks += drain_queue_items(result_queue, merge_fn)
        side_drained_counts = drain_side_queue_items(side_queues)
        raise_if_worker_failed(tasks, stage_label)
        if alive_workers == 0:
            break
        if progress_logger is not None:
            fields = {
                'alive_workers': alive_workers,
                'finished_workers': len(tasks) - alive_workers,
                'merged_result_chunks': merged_result_chunks,
            }
            for label, count in side_drained_counts.items():
                fields[f"drained_{label}"] = count
            progress_logger(**fields)
        time.sleep(poll_interval)
    for task in tasks:
        task.join()
    merged_result_chunks += drain_queue_items(result_queue, merge_fn)
    drain_side_queue_items(side_queues)
    raise_if_worker_failed(tasks, stage_label)

    return {'merged_result_chunks': merged_result_chunks, 'worker_count': len(tasks)}

#################################################################
#                Predicates and Functionalities                 #
#################################################################

def initializePredicateSubsumption(predicates1, predicates2, pred2superPred12={}, pred2superPred21={}, relinit=0.1):
    """ 
    Initializes identical relations to 1.0, all others as given or else to RELINC.

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

class CompactDenseDefaultPredicateMapping:
    """ID-level equivalent of initializePredicateSubsumption."""
    def __init__(self, kb_src, kb_dst, relinit=0.1):
        self.relinit = relinit 
        self.target_predicate_ids = tuple(kb_dst.iter_predicate_ids()) # all target predicate IDs
        dst_predicate_ids = {}
        for pred2_id in kb_dst.iter_predicate_ids():
            pred2 = kb_dst.predicate_for_id(pred2_id)
            dst_predicate_ids.setdefault(pred2, []).append(pred2_id)

        equivalent_pairs = set()
        for pred1_id in kb_src.iter_predicate_ids():
            pred1 = kb_src.predicate_for_id(pred1_id)
            for pred2_id in dst_predicate_ids.get(pred1, ()):
                equivalent_pairs.add((pred1_id, pred2_id))
        self.equivalent_pairs = frozenset(equivalent_pairs) # exact same predicate ID pairs
        self.seed_scores = {}

    def set_score_ids(self, predicate1_id, predicate2_id, score):
        self.seed_scores[(int(predicate1_id), int(predicate2_id))] = float(score)

    def score_ids(self, predicate1_id, predicate2_id):
        pair = (int(predicate1_id), int(predicate2_id))
        if pair in self.seed_scores:
            return self.seed_scores[pair]
        if pair in self.equivalent_pairs:
            return 1.0
        return self.relinit

def is_dense_default_predicate_mapping(mapping):
    return isinstance(mapping, CompactDenseDefaultPredicateMapping)

def initializePredicateIdentityOnly(predicates1, predicates2):
    """
    Initialize predicate subsumption with exact predicate matches only.

    Parameters
    ----------
    predicates1 : set
        Set of predicates in KB1
    predicates2 : set
        Set of predicates in KB2
    
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
    return result

def initializePredicateIdentityOnlyIds(kb_src, kb_dst):
    """ID-level equivalent of initializePredicateIdentityOnly."""
    src_side = side_keys.graph_predicate_side(kb_src, side_keys.PRED1)
    dst_side = side_keys.graph_predicate_side(kb_dst, side_keys.PRED2)
    dst_predicate_ids = {}
    for pred2_id in kb_dst.iter_predicate_ids():
        pred2 = kb_dst.predicate_for_id(pred2_id)
        dst_predicate_ids.setdefault(pred2, []).append(pred2_id)

    result = {}
    for pred1_id in kb_src.iter_predicate_ids():
        pred1 = kb_src.predicate_for_id(pred1_id)
        pred1_key = side_keys.side_predicate_key(src_side, pred1_id)
        equivalent_pred2_ids = set(dst_predicate_ids.get(pred1, ()))
        for pred2_id in equivalent_pred2_ids:
            pred2_key = side_keys.side_predicate_key(dst_side, pred2_id)
            result.setdefault(pred1_key, {})[pred2_key] = 1.0
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

def computeFunctionalitiesIds(kb, gram=[]):
    """
    ID-level functionality computation for CompactGraph.
    Returned keys are local predicate IDs, or tuples of local predicate IDs.
    """
    predicate2numFacts = {}
    predicate2subjects = {}
    for subject_id in range(kb.num_entities()):
        facts = list(kb.triplesWithSubjectIds(subject_id))
        for n in gram:
            if n == 1:
                for fact in facts:
                    predicate_key = fact[PRED]
                    if predicate_key not in predicate2numFacts:
                        predicate2numFacts[predicate_key] = 0
                        predicate2subjects[predicate_key] = set()
                    predicate2numFacts[predicate_key] += 1
                    predicate2subjects[predicate_key].add(fact[SUBJ])
                continue

            cnt = 0
            for evs in combinations(facts, n):
                cnt += 1
                predicate_ids = []
                skip = False
                for fact in evs:
                    inverse_predicate_id = kb._inverse_predicate_id(fact[PRED])
                    if inverse_predicate_id is None:
                        skip = True
                        break
                    predicate_ids.append(inverse_predicate_id)
                if skip:
                    continue
                predicate_key = tuple(sorted(predicate_ids))
                subject_key = tuple(sorted(fact[OBJ] for fact in evs))
                if predicate_key not in predicate2numFacts:
                    predicate2numFacts[predicate_key] = 0
                    predicate2subjects[predicate_key] = set()
                predicate2numFacts[predicate_key] += 1
                predicate2subjects[predicate_key].add(subject_key)
                if cnt > 100000: # avoid memory overflow
                    break
    return {predicate: len(predicate2subjects[predicate]) / predicate2numFacts[predicate] for predicate in predicate2numFacts}

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
#                     Implication Functions                     #
#################################################################

def updateScoreMin(mapping, key1, key2, *body, min_score=0.0):
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
    min_score : float, optional
        Ignore updates whose resulting score is not strictly above this threshold.
    """
    curScore = 0
    key2scores = mapping.get(key1)
    created_pair = key2scores is None or key2 not in key2scores
    if key2scores is not None and key2 in key2scores:
        curScore = key2scores[key2]
    tmp = max(curScore, min(min(body),1.0))

    if tmp >= min_score:
        if key1 not in mapping:
            mapping[key1]={}
        mapping[key1][key2] = tmp
        return created_pair
    return False

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
#                      Score Merge Helpers                      #
#################################################################

def _nested_max_score(ent_max_assign, entity, ent_max_score_cache=None):
    """
    Return the maximum alignment score for an entity from a nested score mapping.

    Parameters
    ----------
    ent_max_assign : dict | None
        Nested dictionary of entity alignment scores.
    entity : hashable
        Entity whose current maximum score should be looked up.
    ent_max_score_cache : dict | None, optional
        Optional cache mapping entity -> maximum similarity score.

    Returns
    -------
    float
        The maximum known alignment score for the entity, or 0.0 if absent.
    """
    if ent_max_assign is None:
        return 0.0
    if ent_max_score_cache is not None and entity in ent_max_score_cache:
        return ent_max_score_cache[entity]
    score = max(ent_max_assign.get(entity, {None: 0.0}).values())
    if ent_max_score_cache is not None:
        ent_max_score_cache[entity] = score
    return score

def _indexed_max_score(max_scores, entity_id):
    """
    Return the maximum alignment score for an entity ID.

    Parameters
    ----------
    max_scores : list | dict | None
        Indexed or dictionary-backed entity maximum scores.
    entity_id : int
        Entity ID whose maximum score should be read.

    Returns
    -------
    float
        The maximum known alignment score for the entity ID, or 0.0 if absent.
    """
    if max_scores is None or entity_id is None:
        return 0.0
    if isinstance(max_scores, dict):
        return max_scores.get(entity_id, 0.0)
    if entity_id < 0 or entity_id >= len(max_scores):
        return 0.0
    return float(max_scores[entity_id])

def merge_same_as_score_chunk(target_scores, score_chunk, min_score=0.0,
                              ent_max_assign=None, ent_max_score_cache=None):
    """
    Merge a chunk of entity scores into the global sameAs mapping using max
    aggregation. Updates are kept only when they improve the existing score 
    for the same source-target pair.

    Parameters
    ----------
    target_scores : dict
        Nested dictionary of global entity alignment scores to update in place.
    score_chunk : dict
        Nested dictionary of worker-produced entity alignment scores.
    min_score : float, optional
        Drop candidate pairs whose score is not strictly above this threshold.
    ent_max_assign : dict | None, optional
        Current bilateral max assignment used to skip pairs that cannot improve
        either endpoint's best known score.
    ent_max_score_cache : dict | None, optional
        Optional cache for maximum scores read from ent_max_assign.

    Returns
    -------
    None
        The target_scores mapping is updated in place.
    """
    for subj1, subj2scores in score_chunk.items():
        target_map = target_scores.get(subj1)
        if target_map is None:
            target_map = {}
            target_scores[subj1] = target_map

        max_score1 = _nested_max_score(ent_max_assign, subj1, ent_max_score_cache)
        for subj2, score in subj2scores.items():
            if score <= min_score:
                continue
            max_score2 = _nested_max_score(ent_max_assign, subj2, ent_max_score_cache)
            if score <= max(max_score1, max_score2):
                continue
            if score > target_map.get(subj2, 0):
                target_map[subj2] = score

        if not target_map:
            target_scores.pop(subj1, None)

def merge_same_as_score_chunk_ids(target_scores, score_chunk, kb_src, kb_dst, min_score=0.0, 
                                  src_max_scores=None, dst_max_scores=None):
    """
    ID-level equivalent of merge_same_as_score_chunk.

    Parameters
    ----------
    target_scores : dict
        Side/id-keyed nested dictionary of global entity alignment scores to
        update in place.
    score_chunk : dict
        Nested dictionary keyed by source entity IDs and target entity IDs.
    kb_src : Graph
        Source knowledge base used to encode source-side entity keys.
    kb_dst : Graph
        Target knowledge base used to encode target-side entity keys.
    min_score : float, optional
        Drop candidate pairs whose score is not strictly above this threshold.
    src_max_scores : list | dict | None, optional
        Current maximum score per source entity ID.
    dst_max_scores : list | dict | None, optional
        Current maximum score per target entity ID.

    Returns
    -------
    None
        The target_scores mapping is updated in place.
    """

    if not side_keys.is_id_keyed_mapping(target_scores):
        raise TypeError("ID-level score chunks require side/id-keyed target_scores.")

    src_side = side_keys.graph_side(kb_src, side_keys.KB1)
    dst_side = side_keys.graph_side(kb_dst, side_keys.KB2)
    for subj1_id, subj2scores in score_chunk.items():
        subj1_key = side_keys.side_entity_key(src_side, subj1_id)
        target_map = target_scores.get(subj1_key)
        if target_map is None:
            target_map = {}
            target_scores[subj1_key] = target_map

        for subj2_id, score in subj2scores.items():
            if score <= min_score:
                continue
            if score <= max(_indexed_max_score(src_max_scores, subj1_id), _indexed_max_score(dst_max_scores, subj2_id)):
                continue
            subj2_key = side_keys.side_entity_key(dst_side, subj2_id)
            if score > target_map.get(subj2_key, 0):
                target_map[subj2_key] = score

        if not target_map:
            target_scores.pop(subj1_key, None)

def _flush_match_scores(result_queue, ent_match_scores):
    if not ent_match_scores:
        return {}, 0
    result_queue.put(ent_match_scores)
    return {}, 0


def computeQuasiEqrel(kb_src, kb_dst, pred2superPred):
    """ 
    Computes the quasi equivalence relations between the two KGs' predicates, 
    the quasi equivalence is represented as r\\cong r' in paper.

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
    # if the predicate mapping is side-keyed (compactGraph)
    if side_keys.is_id_keyed_mapping(pred2superPred):
        src_side = side_keys.graph_predicate_side(kb_src, side_keys.PRED1)
        dst_side = side_keys.graph_predicate_side(kb_dst, side_keys.PRED2)
        quasiEqrel_ = {} # from kb1 to kb2
        for pred1 in pred2superPred:
            if not side_keys.is_side_key(pred1, src_side):
                continue
            for pred2 in pred2superPred[pred1]:
                if not side_keys.is_side_key(pred2, dst_side):
                    continue
                value = max(pred2superPred[pred1][pred2], pred2superPred.get(pred2, {}).get(pred1, 0))
                if pred1 not in quasiEqrel_:
                    quasiEqrel_[pred1] = {}
                quasiEqrel_[pred1][pred2] = value
        return quasiEqrel_

    # if the predicate mapping is directly predicate-keyed
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


#################################################################
#                 Bilateral Entity Assignment                   #
#################################################################

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
    # get max match for kb1
    max_match_e = {}
    # get max match for kb2 
    max_match_e_prime = {}

    for e, e_prime_scores in sameASscore.items():
        if not e_prime_scores:
            continue

        max_score = max(e_prime_scores.values())
        max_targets_e = []
        for e_prime, score in e_prime_scores.items():
            if score == max_score:
                max_targets_e.append(e_prime)

            current_match = max_match_e_prime.get(e_prime)
            if current_match is None or score > current_match['score']:
                max_match_e_prime[e_prime] = {'score': score, 'targets': {e}}
            elif score == current_match['score']:
                current_match['targets'].add(e)

        if max_targets_e:
            max_match_e[e] = {'score': max_score, 'targets': set(max_targets_e)}
    
    # bilateral max assignment
    res_max_assign = {}
    for e, match_data in max_match_e.items():
        max_score = match_data['score']
        max_targets = match_data['targets']

        for e_prime in max_targets:
            if (e_prime in max_match_e_prime and \
                max_match_e_prime[e_prime]['score'] == max_score and \
                e in max_match_e_prime[e_prime]['targets']):

                # write (e, e')
                if e not in res_max_assign:
                    res_max_assign[e] = {}
                res_max_assign[e][e_prime] = max_score
                # write (e',e)
                if e_prime not in res_max_assign:
                    res_max_assign[e_prime] = {}
                res_max_assign[e_prime][e] = max_score
    return res_max_assign

def _has_compact_id_match_api(kb):
    return (hasattr(kb, 'triplesWithSubjectIds') and hasattr(kb, 'localFunctionalityIds'))

def _can_use_compact_id_match(kb_src, kb_dst):
    return _has_compact_id_match_api(kb_src) and _has_compact_id_match_api(kb_dst)

def _compact_entity_count(graph):
    literal_flags = getattr(graph, '_entity_is_literal', None) # compact graph
    if literal_flags is not None:
        return len(literal_flags)
    return graph.entity_count()

class CompactEntityAssignIndex:
    """
    Read-only CSR-like view over bilateral max assignment scores.
    The parent process builds the arrays once as .npy memmaps, 
    and each worker opens the same files read-only to save memory.
    """

    def __init__(self, metadata):
        self.metadata = metadata
        self.src_offsets = utils.np.load(metadata['src_offsets_path'], mmap_mode='r') # CSR row offsets: source ID -> offset into dst_ids.
        self.dst_ids = utils.np.load(metadata['dst_ids_path'], mmap_mode='r') # target entity IDs for all source-to-target assignments.
        self.scores = utils.np.load(metadata['scores_path'], mmap_mode='r') # assignment scores for all source-to-target assignments.
        self.src_max = utils.np.load(metadata['src_max_path'], mmap_mode='r') # best assignment score for each source entity ID.
        self.dst_max = utils.np.load(metadata['dst_max_path'], mmap_mode='r') # best assignment score for each target entity ID.

    @property
    def pair_count(self):
        """Return the total number of source->target assignment pairs in the index."""
        return int(self.metadata.get('pair_count', len(self.dst_ids)))

    def max_src(self, entity_id):
        """Return the best assignment score for a source entity ID."""
        if entity_id is None or entity_id < 0 or entity_id >= len(self.src_max):
            return 0.0
        return float(self.src_max[entity_id])

    def max_dst(self, entity_id):
        """Return the best assignment score for a target entity ID."""
        if entity_id is None or entity_id < 0 or entity_id >= len(self.dst_max):
            return 0.0
        return float(self.dst_max[entity_id])

    def has_src_score(self, entity_id):
        """Return True if the source entity ID has at least one assignment with a positive score."""
        return self.max_src(entity_id) > 0.0

    def has_dst_score(self, entity_id):
        """Return True if the target entity ID has at least one assignment with a positive score."""
        return self.max_dst(entity_id) > 0.0

    def target_items(self, src_id):
        """Return a tuple of (target_id, score) pairs for all target entities currently aligned to the given source entity ID."""
        if src_id is None or src_id < 0 or src_id + 1 >= len(self.src_offsets):
            return ()
        # CSR slice for all target entities currently aligned to this source.
        start = int(self.src_offsets[src_id])
        end = int(self.src_offsets[src_id + 1])
        if start == end:
            return ()
        return tuple((int(self.dst_ids[index]), float(self.scores[index])) for index in range(start, end))

    def score(self, src_id, dst_id):
        """Return the assignment score for a given source and target entity ID pair."""
        if src_id is None or dst_id is None or src_id < 0 or src_id + 1 >= len(self.src_offsets):
            return 0.0
        start = int(self.src_offsets[src_id])
        end = int(self.src_offsets[src_id + 1])
        for index in range(start, end):
            if int(self.dst_ids[index]) == dst_id:
                return float(self.scores[index])
        return 0.0

def build_compact_entity_assign_index(kb_src, kb_dst, ent_max_assign, mmap_dir):
    """
    Build shared array files for compact entity assignment lookups.

    Parameters
    ----------
    kb_src : Graph
        Source knowledge base.
    kb_dst : Graph
        Target knowledge base.
    ent_max_assign : dict
        Current bilateral max assignment of entities.
    mmap_dir : str
        Directory where the .npy memmap arrays should be written.

    Returns
    -------
    dict
        Metadata containing memmap paths, entity counts, pair count, and total
        array bytes. Workers pass this metadata to CompactEntityAssignIndex.
    """
    np = utils.np
    # Rebuild the index every iteration because ent_max_assign changes as newly inferred sameAs scores are merged.
    if os.path.isdir(mmap_dir):
        shutil.rmtree(mmap_dir)
    os.makedirs(mmap_dir, exist_ok=True)

    src_entity_count = _compact_entity_count(kb_src)
    dst_entity_count = _compact_entity_count(kb_dst)
    id_dtype = np.int64 if max(src_entity_count, dst_entity_count) > np.iinfo(np.int32).max else np.int32
    offset_dtype = np.int64
    score_dtype = np.float32

    src_offsets = np.zeros(src_entity_count + 1, dtype=offset_dtype)
    src_max = np.zeros(src_entity_count, dtype=score_dtype)
    dst_max = np.zeros(dst_entity_count, dtype=score_dtype)

    if not side_keys.is_id_keyed_mapping(ent_max_assign):
        raise TypeError("Compact entity assignment index requires side/id-keyed ent_max_assign.")
    src_side = side_keys.graph_side(kb_src, side_keys.KB1)
    dst_side = side_keys.graph_side(kb_dst, side_keys.KB2)

    def scan_assignments(write_arrays=False, cursor=None, dst_ids=None, scores=None):
        """Scan ent_max_assign."""
        for entity_key, target_scores in ent_max_assign.items():
            if not target_scores:
                continue
            side, entity_id = entity_key
            max_score = max(target_scores.values())
            if side == src_side:
                if 0 <= entity_id < src_entity_count:
                    src_max[entity_id] = max(src_max[entity_id], max_score)
                for target_key, score in target_scores.items():
                    if not side_keys.is_side_key(target_key, dst_side):
                        continue
                    target_id = target_key[1]
                    if not (0 <= entity_id < src_entity_count and 0 <= target_id < dst_entity_count):
                        continue
                    if write_arrays:
                        index = cursor[entity_id]
                        dst_ids[index] = target_id
                        scores[index] = score
                        cursor[entity_id] += 1
                    else:
                        src_offsets[entity_id + 1] += 1
            elif side == dst_side and 0 <= entity_id < dst_entity_count:
                dst_max[entity_id] = max(dst_max[entity_id], max_score)

    # First pass: count outgoing source->target pairs and record max-score arrays.
    scan_assignments(write_arrays=False)
    # Convert per-source pair counts into CSR offsets. The final offset is the
    # total number of source->target score pairs that need storage.
    np.cumsum(src_offsets, out=src_offsets)
    pair_count = int(src_offsets[-1])

    paths = {
        'src_offsets_path': os.path.join(mmap_dir, 'src_offsets.npy'),
        'dst_ids_path': os.path.join(mmap_dir, 'dst_ids.npy'),
        'scores_path': os.path.join(mmap_dir, 'scores.npy'),
        'src_max_path': os.path.join(mmap_dir, 'src_max.npy'),
        'dst_max_path': os.path.join(mmap_dir, 'dst_max.npy'),
    }
    array_bytes = 0

    def write_array(path, array):
        nonlocal array_bytes
        mmap_array = np.lib.format.open_memmap(path, mode='w+', dtype=array.dtype, shape=array.shape)
        mmap_array[:] = array
        mmap_array.flush()
        array_bytes += int(array.nbytes)

    write_array(paths['src_offsets_path'], src_offsets)
    write_array(paths['src_max_path'], src_max)
    write_array(paths['dst_max_path'], dst_max)

    dst_ids = np.lib.format.open_memmap(paths['dst_ids_path'], mode='w+', dtype=id_dtype, shape=(pair_count,))
    scores = np.lib.format.open_memmap(paths['scores_path'], mode='w+', dtype=score_dtype, shape=(pair_count,))
    cursor = src_offsets[:-1].copy()
    # Second pass: cursor advances within each source slice while filling sparse target IDs and scores.
    scan_assignments(write_arrays=True, cursor=cursor, dst_ids=dst_ids, scores=scores)
    dst_ids.flush()
    scores.flush()
    array_bytes += int(dst_ids.nbytes) + int(scores.nbytes)

    metadata = {
        **paths,
        'pair_count': pair_count,
        'src_entity_count': src_entity_count,
        'dst_entity_count': dst_entity_count,
        'array_bytes': array_bytes,
    }
    logging.debug(
        "Compact entity assignment array index built | pairs=%s | src_entities=%s | "
        "dst_entities=%s | array_bytes=%.2fMB | dir=%s",
        pair_count, src_entity_count,
        dst_entity_count, array_bytes / (1024 * 1024), mmap_dir,
    )
    return metadata