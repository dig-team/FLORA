"""
This file is part of FLORA, an unsupervised system for automatic knowledge graph (KG) alignment. 
The file is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0) by Yiwen Peng, Thomas Bonald, Fabian Suchanek and Lingyun Huang.

Description: Predicate subrelation mapping utilities, including chunked and compact-graph implementations for updating relation scores.
"""

from collections import OrderedDict
import multiprocessing as mp
import logging
import alignment_base
import log
import side_keys


#################################################################
#                      Subrelation Mapping                      #
#################################################################

def iter_fact_chunks(graph, chunk_size=alignment_base.SUBRELATION_FACT_CHUNK_SIZE):
    """Yield graph facts in bounded chunks."""
    chunk = []
    for fact in graph:
        chunk.append(fact)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def iter_fact_id_chunks(graph, chunk_size=alignment_base.SUBRELATION_FACT_CHUNK_SIZE):
    """Yield ID-level graph facts in bounded chunks."""
    chunk = []
    for fact in graph.iterFactIds():
        chunk.append(fact)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def iter_candidate_fact_id_chunks(graph, active_entity_ids, chunk_size=alignment_base.SUBRELATION_FACT_CHUNK_SIZE):
    """
    Yield ID fact chunks whose subject and object can both match.

    Parameters
    ----------
    graph : Graph
        Knowledge base exposing ID-level subject and fact iteration.
    active_entity_ids : set
        Entity IDs that currently have at least one candidate alignment.
    chunk_size : int, optional
        Maximum number of candidate ID facts per yielded chunk.

    Yields
    ------
    list
        A list of ID-level facts whose subject and object are active entities.
    """
    chunk = []
    for subject_id in sorted(active_entity_ids):
        for fact in graph.triplesWithSubjectIds(subject_id):
            if fact[alignment_base.OBJ] not in active_entity_ids:
                continue
            chunk.append(fact)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk

def merge_additive_score_mapping(target_mapping, score_mapping):
    """Merge a mapping of additive scores into a target mapping, clamping to [0, 1]"""
    for key1, key2scores in score_mapping.items():
        target_scores = target_mapping.setdefault(key1, {})
        for key2, score in key2scores.items():
            target_scores[key2] = max(min(target_scores.get(key2, 0) + score, 1.0), 0)

def _subrelation_update_additive(mapping, key1, key2, value):
    """Update a mapping of additive scores, clamping to [0, 1]"""
    if value <= 0:
        return
    key2scores = mapping.setdefault(key1, {})
    key2scores[key2] = max(min(key2scores.get(key2, 0) + value, 1.0), 0)

def _subject_object_predicate_map_for_subject(kb_lookup, subject_id, cache):
    """
    Return an object-to-predicates map for one target subject using cache.

    Parameters
    ----------
    kb_lookup : Graph
        Target knowledge base used for subject fact lookups.
    subject_id : int
        Target subject entity ID to inspect.
    cache : OrderedDict
        LRU cache from subject ID to object_id -> tuple(predicate_ids).

    Returns
    -------
    tuple
        (object_to_predicates, None) when the subject is cached or cacheable;
        otherwise (None, triples) for low-degree subjects that should be scanned directly.
    """
    cached = cache.get(subject_id)
    # if the subject_id is already cached, return the cached mapping and update the cache
    if cached is not None:
        cache.move_to_end(subject_id)
        return cached, None
    
    # if the subject_id is not cached, look up the triples and build the mapping
    triples = list(kb_lookup.triplesWithSubjectIds(subject_id))
    # if the number of triples is below the threshold, bypass caching 
    if len(triples) < alignment_base.SUBRELATION_SUBJECT_OBJECT_CACHE_DEGREE_THRESHOLD:
        return None, triples
    # if the number of triples is above the threshold, build the mapping and cache it
    object_to_predicates = {}
    for _, predicate_id, object_id in triples:
        object_to_predicates.setdefault(object_id, []).append(predicate_id)
    object_to_predicates = {object_id: tuple(predicate_ids) for object_id, predicate_ids in object_to_predicates.items()} # convert lists to tuples for memory efficiency
    cache[subject_id] = object_to_predicates

    # if the cache exceeds the maximum number of subjects, evict the oldest entry
    if len(cache) > alignment_base.SUBRELATION_SUBJECT_OBJECT_CACHE_MAX_SUBJECTS:
        cache.popitem(last=False)
    return object_to_predicates, None

def _compute_subrelation_chunk(fact_chunk, kb_lookup, ent_max_assign, source_predicate_counts, alpha):
    """
    Compute source-to-target predicate subsumption scores for a chunk of facts.

    Parameters
    ----------
    fact_chunk : list
        Source facts to process.
    kb_lookup : Graph
        Target knowledge base used to find aligned facts.
    ent_max_assign : dict
        Current entity alignment scores.
    source_predicate_counts : dict
        Number of source facts for each source predicate.
    alpha : float
        Benefit of the doubt parameter for additive subrelation scoring.

    Returns
    -------
    tuple
        A local predicate score mapping, number of processed facts, and number
        of facts whose subject and object both had alignment candidates.
    """
    local_mapping = {}
    processed_facts = 0
    matched_facts = 0

    for fact in fact_chunk:
        processed_facts += 1
        subject_scores = ent_max_assign.get(fact[alignment_base.SUBJ])
        if not subject_scores:
            continue
        object_scores = ent_max_assign.get(fact[alignment_base.OBJ])
        if not object_scores:
            continue
        matched_facts += 1

        rel_max_scores = {}
        for aligned_subject, subject_score in subject_scores.items():
            for aligned_predicate, aligned_objects in kb_lookup.subject_items(aligned_subject):
                for aligned_object in aligned_objects:
                    object_score = object_scores.get(aligned_object)
                    if object_score is None:
                        continue
                    score = min(subject_score, object_score)
                    if score > rel_max_scores.get(aligned_predicate, 0): # keep the max score for each aligned predicate
                        rel_max_scores[aligned_predicate] = score

        predicate_weight = alpha / source_predicate_counts.get(fact[alignment_base.PRED], 1)
        for aligned_predicate, score in rel_max_scores.items():
            _subrelation_update_additive(local_mapping, fact[alignment_base.PRED], aligned_predicate, score * predicate_weight)

    return local_mapping, processed_facts, matched_facts

def _compute_subrelation_chunk_compact(fact_chunk, kb_lookup, ent_max_assign_ids,
                                       source_predicate_counts_ids, alpha, subject_object_cache=None):
    """
    Compute ID-level source-to-target predicate subsumption scores for a fact chunk.

    Parameters
    ----------
    fact_chunk : list
        Source ID facts whose subject and object both have alignment candidates.
    kb_lookup : Graph
        Target knowledge base used to find aligned ID facts.
    ent_max_assign_ids : dict
        Current source-ID to target-ID entity alignment scores.
    source_predicate_counts_ids : dict
        Number of source facts for each source predicate ID.
    alpha : float
        Benefit of the doubt parameter for additive subrelation scoring.
    subject_object_cache : OrderedDict | None, optional
        Worker-local cache for high-degree target subject lookups.

    Returns
    -------
    tuple
        A local predicate-ID score mapping and number of processed candidate facts.
    """
    local_mapping = {}
    processed_facts = 0
    # use cache for subject-object-predicate mapping to avoid repeated lookups for high-degree subjects
    if subject_object_cache is None:
        subject_object_cache = OrderedDict()

    for subject_id, predicate_id, object_id in fact_chunk:
        processed_facts += 1
        subject_scores = ent_max_assign_ids[subject_id]
        object_scores = ent_max_assign_ids[object_id]

        rel_max_scores = {}
        for aligned_subject_id, subject_score in subject_scores.items():
            if not kb_lookup.has_subject_id(aligned_subject_id):
                continue
            object_to_predicates, fallback_triples = _subject_object_predicate_map_for_subject(
                kb_lookup,
                aligned_subject_id,
                subject_object_cache,
            )
            if object_to_predicates is None:
                for _, aligned_predicate_id, aligned_object_id in fallback_triples:
                    object_score = object_scores.get(aligned_object_id)
                    if object_score is None:
                        continue
                    score = min(subject_score, object_score)
                    if score > rel_max_scores.get(aligned_predicate_id, 0):
                        rel_max_scores[aligned_predicate_id] = score
            else:
                for aligned_object_id, object_score in object_scores.items():
                    for aligned_predicate_id in object_to_predicates.get(aligned_object_id, ()):
                        score = min(subject_score, object_score)
                        if score > rel_max_scores.get(aligned_predicate_id, 0):
                            rel_max_scores[aligned_predicate_id] = score

        predicate_weight = alpha / source_predicate_counts_ids.get(predicate_id, 1)
        for aligned_predicate_id, score in rel_max_scores.items():
            _subrelation_update_additive(
                local_mapping,
                predicate_id,
                aligned_predicate_id,
                score * predicate_weight,
            )

    return local_mapping, processed_facts

def _map_subrelations_worker(kb_lookup, ent_max_assign, source_predicate_counts, alpha, task_queue, result_queue):
    while True:
        item = task_queue.get()
        if item is None:
            result_queue.put(None)
            return
        chunk_index, fact_chunk = item
        mapping, processed_facts, matched_facts = _compute_subrelation_chunk(
            fact_chunk,
            kb_lookup,
            ent_max_assign,
            source_predicate_counts,
            alpha,
        )
        result_queue.put((chunk_index, mapping, processed_facts, matched_facts))

def _map_subrelations_worker_compact(kb_lookup, ent_max_assign_ids, source_predicate_counts_ids, alpha, task_queue, result_queue):
    subject_object_cache = OrderedDict()
    while True:
        item = task_queue.get()
        if item is None:
            result_queue.put(None)
            return
        chunk_index, fact_chunk = item
        mapping, processed_facts = _compute_subrelation_chunk_compact(
            fact_chunk,
            kb_lookup,
            ent_max_assign_ids,
            source_predicate_counts_ids,
            alpha,
            subject_object_cache=subject_object_cache,
        )
        result_queue.put((chunk_index, mapping, processed_facts))

def _map_subrelations_direction(alpha, facts_iterable, kb_lookup, ent_max_assign,
                                source_predicate_counts, source_total_facts,
                                direction, num_workers, stage_label=None,
                                progress_interval=alignment_base.PROGRESS_LOG_INTERVAL):
    """Map subrelations from a source graph to a target graph in the given direction."""
    task_queue = mp.Queue(maxsize=max(4, num_workers * 2))
    result_queue = mp.Queue(maxsize=max(4, num_workers * 2))
    tasks = []
    mapping = {}
    processed_facts = 0
    matched_facts = 0
    submitted_chunks = 0
    finished_workers = 0
    progress_logger = log._build_progress_logger(stage_label, interval=progress_interval)

    try:
        for _ in range(num_workers):
            task = mp.Process(
                target=_map_subrelations_worker,
                args=(
                    kb_lookup,
                    ent_max_assign,
                    source_predicate_counts,
                    alpha,
                    task_queue,
                    result_queue,
                ),
            )
            task.start()
            tasks.append(task)

        def merge_result(result):
            nonlocal finished_workers, processed_facts, matched_facts
            if result is None:
                finished_workers += 1
                return
            _, chunk_mapping, chunk_processed, chunk_matched = result
            merge_additive_score_mapping(mapping, chunk_mapping)
            processed_facts += chunk_processed
            matched_facts += chunk_matched

        for fact_chunk in iter_fact_chunks(facts_iterable):
            alignment_base._put_task_with_drain(
                task_queue,
                (submitted_chunks, fact_chunk),
                result_queue=result_queue,
                merge_fn=merge_result,
                progress_logger=progress_logger,
                progress_fields={
                    'direction': direction,
                    'submitted_chunks': submitted_chunks,
                    'processed_facts': f"{processed_facts}/{source_total_facts}",
                    'finished_workers': finished_workers,
                },
                worker_tasks=tasks,
                stage_label=stage_label,
            )
            submitted_chunks += 1
            alignment_base.drain_queue_items(result_queue, merge_result)
            if progress_logger is not None:
                progress_logger(
                    direction=direction,
                    submitted_chunks=submitted_chunks,
                    processed_facts=f"{processed_facts}/{source_total_facts}",
                    finished_workers=finished_workers,
                )

        for _ in range(num_workers):
            alignment_base._put_task_with_drain(
                task_queue,
                None,
                result_queue=result_queue,
                merge_fn=merge_result,
                worker_tasks=tasks,
                stage_label=stage_label,
            )

        alignment_base.wait_for_workers_and_drain(
            tasks,
            result_queue,
            merge_result,
            stage_label=stage_label,
            progress_interval=progress_interval,
        )
        return mapping
    finally:
        alignment_base.stop_workers(tasks)
        task_queue.close()
        task_queue.join_thread()
        result_queue.close()
        result_queue.join_thread()

def _can_use_compact_subrelation(kb_src, kb_dst):
    return (
        hasattr(kb_src, 'iterFactIds')
        and hasattr(kb_src, 'predicate_id')
        and hasattr(kb_src, 'predicate_for_id')
        and hasattr(kb_dst, 'iterFactIds')
        and hasattr(kb_dst, 'predicate_for_id')
    )

def _encode_compact_predicate_counts(graph):
    if hasattr(graph, 'predicate_counts_ids'):
        return graph.predicate_counts_ids()
    encoded_counts = {}
    for predicate, count in graph.predicates().items():
        predicate_id = graph.predicate_id(predicate)
        if predicate_id is not None:
            encoded_counts[predicate_id] = count
    return encoded_counts

def _map_subrelations_direction_compact(alpha, kb_src, kb_dst, ent_max_assign_ids,
                                        source_predicate_counts_ids, source_total_facts,
                                        direction, num_workers, stage_label=None,
                                        progress_interval=alignment_base.PROGRESS_LOG_INTERVAL):
    """ID-level equivalent of _map_subrelations_direction."""
    task_queue = mp.Queue(maxsize=max(4, num_workers * 2))
    result_queue = mp.Queue(maxsize=max(4, num_workers * 2))
    tasks = []
    mapping = {}
    processed_facts = 0
    matched_facts = 0
    submitted_chunks = 0
    finished_workers = 0
    progress_logger = log._build_progress_logger(stage_label, interval=progress_interval)

    try:
        for _ in range(num_workers):
            task = mp.Process(
                target=_map_subrelations_worker_compact,
                args=(
                    kb_dst,
                    ent_max_assign_ids,
                    source_predicate_counts_ids,
                    alpha,
                    task_queue,
                    result_queue,
                ),
            )
            task.start()
            tasks.append(task)

        def merge_result(result):
            nonlocal finished_workers, processed_facts, matched_facts
            if result is None:
                finished_workers += 1
                return
            _, chunk_mapping, chunk_processed = result
            merge_additive_score_mapping(mapping, chunk_mapping)
            processed_facts += chunk_processed
            logging.debug(
                "%s compact subrelation direct worker result | direction=%s | "
                "processed_facts=%s",
                stage_label or "Subrelation stage",
                direction,
                chunk_processed,
            )

        for fact_chunk in iter_candidate_fact_id_chunks(kb_src, ent_max_assign_ids.keys()):
            alignment_base._put_task_with_drain(
                task_queue,
                (submitted_chunks, fact_chunk),
                result_queue=result_queue,
                merge_fn=merge_result,
                progress_logger=progress_logger,
                progress_fields={
                    'direction': direction,
                    'submitted_chunks': submitted_chunks,
                    'processed_facts': f"{processed_facts}/{source_total_facts}",
                    'matched_facts': matched_facts,
                    'finished_workers': finished_workers,
                },
                worker_tasks=tasks,
                stage_label=stage_label,
            )
            submitted_chunks += 1
            alignment_base.drain_queue_items(result_queue, merge_result)
            if progress_logger is not None:
                progress_logger(
                    direction=direction,
                    submitted_chunks=submitted_chunks,
                    processed_facts=f"{processed_facts}/{source_total_facts}",
                    matched_facts=matched_facts,
                    finished_workers=finished_workers,
                )

        for _ in range(num_workers):
            alignment_base._put_task_with_drain(
                task_queue,
                None,
                result_queue=result_queue,
                merge_fn=merge_result,
                worker_tasks=tasks,
                stage_label=stage_label,
            )

        alignment_base.wait_for_workers_and_drain(
            tasks,
            result_queue,
            merge_result,
            stage_label=stage_label,
            progress_interval=progress_interval,
        )
        logging.info(
            "%s compact subrelation direct stats | direction=%s | "
            "submitted_chunks=%s | processed_facts=%s/%s",
            stage_label or "Subrelation stage", direction,
            submitted_chunks, processed_facts, source_total_facts,
        )
        return mapping
    finally:
        alignment_base.stop_workers(tasks)
        task_queue.close()
        task_queue.join_thread()
        result_queue.close()
        result_queue.join_thread()

def map_subrelations(alpha, kb_src, kb_dst, ent_maxAssign, previouspredicate2superPredicate,
                     stage_label=None, progress_interval=alignment_base.PROGRESS_LOG_INTERVAL, num_workers=1):
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
    stage_label : str | None, optional
        Label used for throttled progress logging during subrelation mapping.
    progress_interval : float, optional
        Minimum interval in seconds between progress log messages.
    num_workers : int, optional
        Number of worker processes to use.

    Returns
    -------
    None
        previouspredicate2superPredicate is updated in place.
    """
    src_predicate_counts = kb_src.predicates()
    dst_predicate_counts = kb_dst.predicates()
    num_workers = max(1, num_workers or mp.cpu_count())
    src_total_facts = sum(src_predicate_counts.values())
    dst_total_facts = sum(dst_predicate_counts.values())
    logging.info("%s worker config | workers=%s ", stage_label or "Subrelation stage", num_workers)

    if _can_use_compact_subrelation(kb_src, kb_dst):
        logging.info("%s implementation | mode=compact", stage_label or "Subrelation stage")
        src_to_dst_scores, _, _ = side_keys.encode_compact_entity_assign(kb_src, kb_dst, ent_maxAssign)
        dst_to_src_scores, _, _ = side_keys.encode_compact_entity_assign(kb_dst, kb_src, ent_maxAssign)
        src_predicate_counts_ids = _encode_compact_predicate_counts(kb_src)
        dst_predicate_counts_ids = _encode_compact_predicate_counts(kb_dst)

        log.prepare_for_worker_fork(stage_label or "Compact subrelation worker-stage")
        # kb1 -> kb2
        pred2superPred1 = _map_subrelations_direction_compact(
            alpha,
            kb_src,
            kb_dst,
            src_to_dst_scores,
            src_predicate_counts_ids,
            src_total_facts,
            'kb1_to_kb2',
            num_workers,
            stage_label=stage_label,
            progress_interval=progress_interval,
        )
        # kb2 -> kb1
        pred2superPred2 = _map_subrelations_direction_compact(
            alpha,
            kb_dst,
            kb_src,
            dst_to_src_scores,
            dst_predicate_counts_ids,
            dst_total_facts,
            'kb2_to_kb1',
            num_workers,
            stage_label=stage_label,
            progress_interval=progress_interval,
        )

        src_side = side_keys.graph_predicate_side(kb_src, side_keys.PRED1)
        dst_side = side_keys.graph_predicate_side(kb_dst, side_keys.PRED2)
        pred2superPred1 = side_keys.side_key_compact_predicate_mapping(
            pred2superPred1,
            src_side,
            dst_side,
        )
        pred2superPred2 = side_keys.side_key_compact_predicate_mapping(
            pred2superPred2,
            dst_side,
            src_side,
        )
        alignment_base.updatePredicateSubsumption(pred2superPred1, pred2superPred2, previouspredicate2superPredicate)
        return

    log.prepare_for_worker_fork(stage_label or "Subrelation worker-stage")
    # kb1 -> kb2
    pred2superPred1 = _map_subrelations_direction(
        alpha,
        kb_src,
        kb_dst,
        ent_maxAssign,
        src_predicate_counts,
        src_total_facts,
        'kb1_to_kb2',
        num_workers,
        stage_label=stage_label,
        progress_interval=progress_interval,
    )
    # kb2 -> kb1
    pred2superPred2 = _map_subrelations_direction(
        alpha,
        kb_dst,
        kb_src,
        ent_maxAssign,
        dst_predicate_counts,
        dst_total_facts,
        'kb2_to_kb1',
        num_workers,
        stage_label=stage_label,
        progress_interval=progress_interval,
    )

    # complete the subrelation mapping
    alignment_base.updatePredicateSubsumption(pred2superPred1, pred2superPred2, previouspredicate2superPredicate)