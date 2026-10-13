"""
This file is part of FLORA, an unsupervised system for automatic knowledge graph (KG) alignment. 
The file is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0) by Yiwen Peng, Thomas Bonald, Fabian Suchanek and Lingyun Huang.

Description: Entity bootstrap and iterative entity matching procedures used by FLORA's fuzzy-logic alignment loop.
"""

from collections import OrderedDict
import multiprocessing as mp
import time
import logging
import shutil
import tempfile

import alignment_base
import log
import utils
import side_keys


#################################################################
#                      Bootstrap Procedure                      #
#################################################################

def _1st_iteration(kb_src, kb_dst, pred2superPred, functionalities,
        queue, ent_match_tuple_queue, ent_max_assign, min_score=0.0):
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
    min_score : float, optional
        Ignore bootstrap matches whose score is not strictly above this threshold.
    """
    ent_match_scores = dict()
    pending_pairs = 0
    while True:
        entity_chunk = queue.get()
        if entity_chunk is None:
            break
        
        # keep only the subjects that are in the max assignment
        for subj_kb1 in entity_chunk:
            subject_scores = ent_max_assign.get(subj_kb1)
            if not subject_scores:
                continue

            for fact1 in kb_src.triplesWithSubject(subj_kb1):
                # We don't match literals
                if utils.isLiteral(fact1[alignment_base.OBJ]):
                    continue

                pred1 = fact1[alignment_base.PRED]
                pred_targets = pred2superPred.get(pred1)
                if not pred_targets:
                    continue

                localfunc1 = kb_src.localFunctionality(subj_kb1, pred1)
                globalfunc1 = functionalities[pred1]
                for subj_kb2, subject_score in subject_scores.items():
                    if not kb_dst.has_subject(subj_kb2):
                        continue
                    for fact2 in kb_dst.triplesWithSubject(subj_kb2, pred_targets):
                        # We don't match literals
                        if utils.isLiteral(fact2[alignment_base.OBJ]):
                            continue

                        pred2 = fact2[alignment_base.PRED]
                        reverse_pred_targets = pred2superPred.get(pred2, {})
                        # Update
                        if alignment_base.updateScoreMin(
                            # Objects are the same, ...
                            ent_match_scores, fact1[alignment_base.OBJ], fact2[alignment_base.OBJ],
                            # ... if the subjects are the same, ...
                            subject_score,
                            # ... and the predicate is locally functional, ...
                            localfunc1, kb_dst.localFunctionality(subj_kb2, pred2),
                            # ... and the predicate is globally functional,
                            globalfunc1, functionalities[pred2],
                            # ... and the target predicate is subsumed.
                            max(pred_targets[pred2], reverse_pred_targets.get(pred1, 0)),
                            # ... and the minimum score threshold is met.
                            min_score=min_score,
                        ):
                            pending_pairs += 1
                        if pending_pairs >= alignment_base.MATCH_RESULT_FLUSH_PAIRS:
                            ent_match_scores, pending_pairs = alignment_base._flush_match_scores(ent_match_tuple_queue, ent_match_scores)
    ent_match_scores, pending_pairs = alignment_base._flush_match_scores(ent_match_tuple_queue, ent_match_scores)
    exit(0)

def _1st_iteration_compact(kb_src, kb_dst, pred2superPred, functionalities,
        queue, ent_match_tuple_queue, ent_max_assign, min_score=0.0,
        src_to_dst_scores=None, entity_assign_metadata=None):
    """ID-level bootstrap worker for CompactGraph."""

    ent_match_scores = dict()
    pending_pairs = 0

    entity_assign_index = (
        alignment_base.CompactEntityAssignIndex(entity_assign_metadata)
        if entity_assign_metadata is not None
        else None
    )
    if src_to_dst_scores is None and entity_assign_index is None:
        src_to_dst_scores, _, _ = side_keys.encode_compact_entity_assign(kb_src, kb_dst, ent_max_assign)
    if not alignment_base.is_dense_default_predicate_mapping(pred2superPred):
        raise TypeError("Compact bootstrap expects CompactDenseDefaultPredicateMapping.")
    compact_predicate_score = pred2superPred.score_ids
    src_functionalities, _, dst_functionalities, _ = side_keys.resolve_compact_worker_functionalities(
        kb_src,
        kb_dst,
        functionalities,
    )

    # Helper function to flush the current match scores to the parent process
    def flush_compact_match_scores():
        nonlocal ent_match_scores, pending_pairs
        if not ent_match_scores:
            ent_match_scores = {}
            pending_pairs = 0
            return
        ent_match_tuple_queue.put(ent_match_scores)
        ent_match_scores = {}
        pending_pairs = 0

    while True:
        entity_chunk = queue.get()
        if entity_chunk is None:
            break

        for subj_kb1 in entity_chunk:
            subj_kb1_id = side_keys.coerce_entity_id(kb_src, subj_kb1)
            if subj_kb1_id is None:
                continue

            if entity_assign_index is not None:
                subject_score_items = entity_assign_index.target_items(subj_kb1_id)
            else:
                subject_scores = src_to_dst_scores.get(subj_kb1_id)
                subject_score_items = () if not subject_scores else tuple(subject_scores.items())
            if not subject_score_items:
                continue

            for _, pred1_id, obj1_id in kb_src.triplesWithSubjectIds(subj_kb1_id):
                if kb_src.is_literal_id(obj1_id):
                    continue

                localfunc1 = kb_src.localFunctionalityIds(subj_kb1_id, pred1_id)
                globalfunc1 = src_functionalities[pred1_id]

                for subj_kb2_id, subject_score in subject_score_items:
                    if not kb_dst.has_subject_id(subj_kb2_id):
                        continue

                    for _, pred2_id, obj2_id in kb_dst.triplesWithSubjectIds(subj_kb2_id):
                        if kb_dst.is_literal_id(obj2_id):
                            continue

                        if alignment_base.updateScoreMin(
                            ent_match_scores, obj1_id, obj2_id,
                            subject_score,
                            localfunc1, kb_dst.localFunctionalityIds(subj_kb2_id, pred2_id),
                            globalfunc1, dst_functionalities[pred2_id],
                            compact_predicate_score(pred1_id, pred2_id),
                            min_score=min_score,
                        ):
                            pending_pairs += 1
                        if pending_pairs >= alignment_base.MATCH_RESULT_FLUSH_PAIRS:
                            flush_compact_match_scores()
    flush_compact_match_scores()
    exit(0)

def bootstrap_algo(kb_src, kb_dst, sameAsScore, pred2superPred, functionalities,
                   min_score=0.0, num_workers=None):
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
    min_score : float, optional
        Ignore bootstrap matches whose score is not strictly above this threshold.
    num_workers : int | None, optional
        Number of worker processes to use. When None, use the CPU count.

    Returns
    -------
    None
        The sameAsScore mapping is updated in place with bootstrap matches.
    """
    stage_name = "Bootstrap worker-stage"
    logging.info("%s worker config | workers=%s ", stage_name , num_workers)
    ent_max_assign = alignment_base.bilateral_max_assign(sameAsScore)
    tasks = []
    ent_queue_ = None
    ent_match_tuple_queue_ = None
    bootstrap_assign_mmap_dir = None
    memory_tracker = None
    try:
        log.prepare_for_worker_fork(stage_name)
        num_workers = max(1, num_workers or mp.cpu_count())
        queue_size = max(4, num_workers * 2)
        ent_queue_ = mp.Queue(maxsize=queue_size)
        ent_match_tuple_queue_ = mp.Queue(maxsize=queue_size)

        use_compact_id_match = alignment_base._can_use_compact_id_match(kb_src, kb_dst)
        worker_target = _1st_iteration_compact if use_compact_id_match else _1st_iteration
        compact_src_to_dst_scores = None
        compact_assign_metadata = None
        compact_assign_index = None
        if use_compact_id_match and not side_keys.is_id_keyed_mapping(sameAsScore):
            raise TypeError(
                "CompactGraph bootstrap requires side/id-keyed sameAsScore; "
                "encode inputs with side_keys.maybe_encode_same_as_scores first."
            )
        if use_compact_id_match:
            bootstrap_assign_mmap_dir = tempfile.mkdtemp(prefix="flora_bootstrap_assign_")
            compact_assign_metadata = alignment_base.build_compact_entity_assign_index(
                kb_src,
                kb_dst,
                ent_max_assign,
                bootstrap_assign_mmap_dir,
            )
            compact_assign_index = alignment_base.CompactEntityAssignIndex(compact_assign_metadata)

        def merge_chunk(ent_match_score_dict):
            if use_compact_id_match:
                alignment_base.merge_same_as_score_chunk_ids(
                    sameAsScore,
                    ent_match_score_dict,
                    kb_src,
                    kb_dst,
                    min_score=min_score,
                    src_max_scores=compact_assign_index.src_max,
                    dst_max_scores=compact_assign_index.dst_max,
                )
            else:
                alignment_base.merge_same_as_score_chunk(sameAsScore, ent_match_score_dict, min_score=min_score)
        
        logging.debug("%s worker implementation | mode=%s", stage_name, "compact-id" if worker_target is _1st_iteration_compact else "string")
        logging.info("%s active source candidates | sources=%s", stage_name, len(ent_max_assign))
        if compact_assign_metadata is not None:
            logging.debug(
                "%s compact source-target assignments | pairs=%s | array_bytes=%.2fMB",
                stage_name,
                compact_assign_metadata['pair_count'],
                compact_assign_metadata['array_bytes'] / (1024 * 1024),
            )

        for _ in range(num_workers):
            args = (
                kb_src, kb_dst,
                pred2superPred,
                functionalities,
                ent_queue_,
                ent_match_tuple_queue_,
                ent_max_assign,
                min_score,
            )
            if use_compact_id_match:
                args = args + (compact_src_to_dst_scores, compact_assign_metadata)
            task = mp.Process(target=worker_target, args=args)
            task.start()
            tasks.append(task)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            memory_tracker = log.ProcessMemoryPeakTracker(stage_name, [task.pid for task in tasks])
            memory_tracker.log_current(f"{stage_name} memory start")
            memory_tracker.start()
        alignment_base.feed_entity_chunks(
            ent_queue_,
            kb_src,
            num_workers,
            result_queue=ent_match_tuple_queue_,
            merge_fn=merge_chunk,
            include_literals=True,
            stage_label=stage_name,
            only_literals=True,
            source_entities=ent_max_assign.keys(),
            worker_tasks=tasks,
        )
        alignment_base.wait_for_workers_and_drain(
            tasks,
            ent_match_tuple_queue_,
            merge_chunk,
            stage_label=stage_name,
        )
        if memory_tracker is not None:
            memory_tracker.log_peak()
    finally:
        if memory_tracker is not None:
            memory_tracker.stop()
        if ent_queue_ is not None:
            ent_queue_.close()
            ent_queue_.join_thread()
        if ent_match_tuple_queue_ is not None:
            ent_match_tuple_queue_.close()
            ent_match_tuple_queue_.join_thread()
        if bootstrap_assign_mmap_dir is not None:
            shutil.rmtree(bootstrap_assign_mmap_dir, ignore_errors=True)


#################################################################
#                    Entity Matching Workers                    #
#################################################################

def _have_same_repetition_pattern(values1, values2):
    """Return whether both sequences repeat values at the same positions."""
    def canonical_pattern(values):
        value_codes = {}
        pattern = []
        for value in values:
            if value not in value_codes:
                value_codes[value] = len(value_codes)
            pattern.append(value_codes[value])
        return pattern
    return canonical_pattern(values1) == canonical_pattern(values2)


def _match_entities_by_rules_compact(kb_src, kb_dst, quasiEqvirel, queue,
                                     ent_match_tuple_queue, ent_max_assign,
                                     functionalities, params, profile_queue=None):
    """
    Match entities using ID-level CompactGraph rules.

    Parameters
    ----------
    kb_src : Graph
        The source knowledge base.
    kb_dst : Graph
        The target knowledge base.
    quasiEqvirel : dict
        Nested dictionary of quasi equivalence scores between predicates.
    queue : mp.Queue
        Multiprocessing queue containing chunks of source entities to process.
    ent_match_tuple_queue : mp.Queue
        Multiprocessing queue receiving nested entity alignment score chunks.
    ent_max_assign : dict
        Current bilateral max assignment of entities.
    functionalities : dict
        Predicate and predicate-group functionality scores.
    params : dict
        Runtime parameters, including gramN, pruning thresholds, and the
        shared compact entity assignment index metadata.
    profile_queue : mp.Queue | None, optional
        Optional queue receiving worker timing and pruning statistics.

    Returns
    -------
    None
        Match scores and optional profile data are emitted through queues.
    """
    ent_match_scores = dict()
    pending_pairs = 0
    profile_stats = {
        'build_facts': 0.0, # seconds spent finding and ranking usable source facts
        'collect_pairs': 0.0, # seconds spent collecting target evidence pairs
        'select': 0.0, # seconds spent selecting target entity candidates
        'align': 0.0, # seconds spent applying entity alignment rules
        'entities_seen': 0, # source entities read from the task queue
        'entities_processed': 0, # non-literal, non-finalized source entities processed
        'skipped_matched': 0, # source entities skipped because they already match strongly
        'fact_upper_bound_pruned_facts': 0, # source facts skipped by best-case score bound
        'target_score_upper_bound_pruned_predicates': 0, # target predicates skipped by score bound
        'target_score_upper_bound_pruned_candidates': 0, # target object candidates losing all predicates to score bound
        'upper_bound_pruned_evidence': 0, # evidence pairs skipped by best-case score bound
        'upper_bound_pruned_rules': 0, # final rule applications skipped by best-case score bound
        'target_hub_functionality_pruned_predicates': 0, # target predicates skipped for high degree
        'target_hub_functionality_pruned_candidates': 0, # target object candidates losing all predicates to high degree
        'total_candidate_obj2': 0, # target object candidates considered across entities
        'total_scanned_evi2': 0, # target evidence triples scanned across entities
        'max_scanned_evi2_entity': 0, # largest scanned target evidence count for one entity
        'max_context_pairs_entity': 0, # largest retained evidence-pair count for one entity
        'max_align_time_entity_s': 0.0, # longest rule-application time for one entity
    }
    # bilateral max assignment
    entity_assign_metadata = params.get('_compact_entity_assign_index')
    if entity_assign_metadata is None:
        raise RuntimeError("Compact entity matching requires shared entity assignment arrays")
    entity_assign_index = alignment_base.CompactEntityAssignIndex(entity_assign_metadata)
    # quasi predicate scores
    quasi_scores, quasi_predicates, max_pred_scores = side_keys.encode_compact_predicate_scores(
        kb_src,
        kb_dst,
        quasiEqvirel,
    )
    # predicate functionalities
    (   src_functionalities,
        src_group_functionalities,
        dst_functionalities,
        dst_group_functionalities,
    ) = side_keys.resolve_compact_worker_functionalities(
        kb_src,
        kb_dst,
        functionalities,
    )
    minimum_output_score = max(0.0, float(params.get('prune_min_score', 0.0) or 0.0))
    target_hub_degree_threshold = params.get('target_hub_degree_threshold', 10000)
    target_hub_degree_threshold = (target_hub_degree_threshold if target_hub_degree_threshold and target_hub_degree_threshold > 0 else None)

    def compact_group_functionality(group_cache, single_predicate_cache, predicate_ids):
        """Get the functionality score for a group of predicates using the group cache."""
        if len(predicate_ids) == 1:
            return single_predicate_cache.get(predicate_ids[0], 1.0)
        return group_cache.get(predicate_ids, 1.0)

    def max_ent_score_src(entity_id):
        """Get the maximum score of any target entity aligned to this source entity."""
        return entity_assign_index.max_src(entity_id)

    def max_ent_score_dst(entity_id):
        """Get the maximum score of any source entity aligned to this target entity."""
        return entity_assign_index.max_dst(entity_id)

    def has_ent_score_src(entity_id):
        """Check if this source entity has any aligned target entities with nonzero score."""
        return entity_assign_index.has_src_score(entity_id)

    def has_ent_score_dst(entity_id):
        """Check if this target entity has any aligned source entities with nonzero score."""
        return entity_assign_index.has_dst_score(entity_id)

    def active_pair_score(src_id, dst_id):
        """Get the current score of a source-target entity pair from the bilateral max assignment."""
        return entity_assign_index.score(src_id, dst_id)

    def active_target_scores(src_id):
        """Get all target entities and their scores currently aligned to this source entity."""
        return entity_assign_index.target_items(src_id)

    def max_pred_score(predicate_id):
        """Get the maximum quasi equivalence score for this predicate across all target predicates."""
        return max_pred_scores.get(predicate_id, 0)

    def current_pair_score(subj1_id, subj2_id):
        """Get the current best score of a source-target entity pair."""
        # alignment scores generated by this worker in current iteration
        generated_scores = ent_match_scores.get(subj1_id) 
        generated_score = 0.0 if generated_scores is None else generated_scores.get(subj2_id, 0.0)
        # alignment scores from the bilateral max assignment
        active_score = active_pair_score(subj1_id, subj2_id)
        return max(generated_score, active_score)

    def rule_score_floor(subj1_id, subj2_id):
        """Get the minimum score floor for a source-target entity pair based on current scores and max entity scores."""
        return max(
            minimum_output_score,
            current_pair_score(subj1_id, subj2_id),
            max_ent_score_src(subj1_id),
            max_ent_score_dst(subj2_id),
        )

    def rule_can_survive(subj1_id, subj2_id, upper_bound):
        """Check if a score can survive the upper bound score pruning."""
        return upper_bound > rule_score_floor(subj1_id, subj2_id)

    def best_case_hmean_with_score(score, max_count):
        """Compute the best-case harmonic mean of a score with a maximum count of evidence."""
        if score <= 0:
            return 0.0
        if score >= 1.0:
            return 1.0
        max_count = max(1, int(max_count))
        return max_count / ((1.0 / score) + max_count - 1) # all other evidence is perfect (score=1.0)

    def fact_can_survive_upper_bound(fact_kb1, quasi_score_map, obj_kb2_scores):
        """Check if a source fact can survive the upper bound score pruning considering the best object and predicate scores."""
        if not quasi_score_map or not obj_kb2_scores:
            return False

        best_obj_score = max((score for _, score in obj_kb2_scores), default=0.0)
        best_pred_score = max(quasi_score_map.values(), default=0.0)
        if best_obj_score <= 0 or best_pred_score <= 0:
            return False
        max_rule_evidence = min(20, max(1, params.get('gramN', 20)))
        fact_upper_bound = min(
            best_case_hmean_with_score(best_obj_score, max_rule_evidence),
            best_case_hmean_with_score(best_pred_score, max_rule_evidence),
        )
        return fact_upper_bound > max(minimum_output_score, max_ent_score_src(fact_kb1[alignment_base.OBJ]))

    def build_predicate_bound_filter(fact_kb1, quasi_score_map, predicate_ids):
        """Precompute predicate-side score bounds for a source fact."""
        floor = max(minimum_output_score, max_ent_score_src(fact_kb1[alignment_base.OBJ]))
        max_rule_evidence = min(20, max(1, params.get('gramN', 20)))
        always_pruned = 0
        surviving_predicates = []
        for predicate_id in predicate_ids:
            pred_bound = best_case_hmean_with_score(
                quasi_score_map[predicate_id],
                max_rule_evidence,
            )
            if pred_bound <= floor:
                always_pruned += 1
            else:
                surviving_predicates.append(predicate_id)
        return floor, max_rule_evidence, always_pruned, tuple(surviving_predicates)

    def filter_target_predicates_by_functionality_bound(fact_kb1, obj_kb2_id, obj_score,
            quasi_score_map, predicate_ids, predicate_bound_filter=None):
        """Filter target predicates based on functionality and score upper bounds."""
        predicates = tuple(predicate_ids)
        hub_pruning_enabled = target_hub_degree_threshold is not None
        # Skip predicates that cannot survive the upper bound score pruning
        if predicate_bound_filter is None: # precompute the predicate-side score bounds for this source fact
            predicate_bound_filter = build_predicate_bound_filter(fact_kb1, quasi_score_map, predicates)
        floor, max_rule_evidence, score_pruned_predicates, predicates = predicate_bound_filter
        obj_bound = best_case_hmean_with_score(obj_score, max_rule_evidence)
        if obj_bound <= floor:
            profile_stats['target_score_upper_bound_pruned_predicates'] += (
                score_pruned_predicates + len(predicates)
            )
            profile_stats['target_score_upper_bound_pruned_candidates'] += 1
            return ()
        if not predicates:
            if score_pruned_predicates:
                profile_stats['target_score_upper_bound_pruned_predicates'] += score_pruned_predicates
                profile_stats['target_score_upper_bound_pruned_candidates'] += 1
            return ()

        # Hub pruning is the only reason to query per subject-predicate counts;
        # nonexistent predicates naturally yield no evidence during scanning.
        if not hub_pruning_enabled:
            if score_pruned_predicates:
                profile_stats['target_score_upper_bound_pruned_predicates'] += score_pruned_predicates
            return predicates
        if not hasattr(kb_dst, '_objects_for_subject_predicate_id_count'):
            return predicates

        kept_predicates = []
        hub_pruned_predicates = 0

        for predicate_id in predicates:
            object_count = kb_dst._objects_for_subject_predicate_id_count(obj_kb2_id, predicate_id)
            if object_count <= 0:
                continue

            if hub_pruning_enabled and object_count >= target_hub_degree_threshold:
                hub_pruned_predicates += 1
                continue

            kept_predicates.append(predicate_id)

        if score_pruned_predicates:
            profile_stats['target_score_upper_bound_pruned_predicates'] += score_pruned_predicates
        if hub_pruned_predicates:
            profile_stats['target_hub_functionality_pruned_predicates'] += hub_pruned_predicates
        if not kept_predicates:
            if score_pruned_predicates:
                profile_stats['target_score_upper_bound_pruned_candidates'] += 1
            if hub_pruned_predicates:
                profile_stats['target_hub_functionality_pruned_candidates'] += 1
        return tuple(kept_predicates)

    def iter_dst_evidence(subject_id, predicate_id_set):
        return kb_dst.triplesWithSubjectIdsFiltered(subject_id, predicate_id_set)

    def flush_compact_match_scores():
        nonlocal ent_match_scores, pending_pairs
        if not ent_match_scores:
            ent_match_scores = {}
            pending_pairs = 0
            return
        ent_match_tuple_queue.put(ent_match_scores)
        ent_match_scores = {}
        pending_pairs = 0

    def add_entity_evidence_pair(context, subj2_id, evi2, fact_kb1, score):
        """Add a source-target entity pair with evidence to the context, keeping only the best score for each target entity."""
        subj2_pairs = context['subj2_pairs']
        pair_map = subj2_pairs.get(subj2_id)
        if pair_map is None:
            pair_map = {}
            subj2_pairs[subj2_id] = pair_map
        previous_pair = pair_map.get(evi2)
        if previous_pair is None or score > previous_pair[1]:
            pair_map[evi2] = (fact_kb1, score)

    def collect_fact_pairs_by_adjacency(
            context,
            fact_kb1,
            quasi_score_map,
            quasi_predicate_ids,
            obj_kb2_scores):
        """ Collect target entity candidates and their evidence for a given source fact.
            Parameters:
                context: The alignment context, including entity memory profile and evidence pair mappings.
                fact_kb1: The source fact.
                quasi_score_map: A map of quasi-scores for predicates.
                quasi_predicate_ids: A list of quasi-predicate IDs.
                obj_kb2_scores: A list of target object scores.
        """
        entity_memory_profile = context['entity_memory_profile']
        tmp_subj2_evi2 = {}
        subj2_maxsubrel_score = {}
        all_quasi_predicate_set = frozenset(quasi_predicate_ids)
        predicate_bound_filter = build_predicate_bound_filter(fact_kb1, quasi_score_map, quasi_predicate_ids)
        for obj_kb2_id, _obj_score in sorted(
            obj_kb2_scores,
            key=lambda item: (-item[1], item[0]),
        ):
            entity_memory_profile['candidate_obj2'] += 1
            aligned_evi2 = []
            maxsubrel_score = 0
            if not kb_dst.has_subject_id(obj_kb2_id):
                continue
            filtered_predicate_ids = filter_target_predicates_by_functionality_bound(
                fact_kb1,
                obj_kb2_id,
                _obj_score,
                quasi_score_map,
                quasi_predicate_ids,
                predicate_bound_filter,
            )
            if not filtered_predicate_ids:
                continue
            evidence_iter = iter_dst_evidence(
                obj_kb2_id,
                all_quasi_predicate_set if len(filtered_predicate_ids) == len(quasi_predicate_ids)
                else frozenset(filtered_predicate_ids),
            )
            for evi2_ in evidence_iter:
                if kb_dst.is_literal_id(evi2_[alignment_base.OBJ]):
                    continue
                entity_memory_profile['scanned_evi2'] += 1
                subrel_score = quasi_score_map[evi2_[alignment_base.PRED]]
                if subrel_score > maxsubrel_score:
                    maxsubrel_score = subrel_score
                    aligned_evi2 = [evi2_]
                if subrel_score == maxsubrel_score:
                    aligned_evi2.append(evi2_)

            if len(aligned_evi2) > entity_memory_profile['aligned_evi2_peak']:
                entity_memory_profile['aligned_evi2_peak'] = len(aligned_evi2)
            for evi2 in aligned_evi2:
                subj2_id = evi2[alignment_base.OBJ]
                subrel_score = quasi_score_map[evi2[alignment_base.PRED]]
                if subrel_score > subj2_maxsubrel_score.get(subj2_id, 0):
                    subj2_maxsubrel_score[subj2_id] = subrel_score
                    tmp_subj2_evi2[subj2_id] = evi2

        if len(tmp_subj2_evi2) > entity_memory_profile['tmp_subj2_peak']:
            entity_memory_profile['tmp_subj2_peak'] = len(tmp_subj2_evi2)
        for subj2_id, single_evi2 in tmp_subj2_evi2.items():
            obj_score = active_pair_score(fact_kb1[alignment_base.SUBJ], single_evi2[alignment_base.SUBJ])
            pred_score = quasi_scores[fact_kb1[alignment_base.PRED]][single_evi2[alignment_base.PRED]]
            score = min(obj_score, pred_score)
            max_rule_evidence = min(20, max(1, params.get('gramN', 20)))
            upper_bound_score = min(
                best_case_hmean_with_score(obj_score, max_rule_evidence),
                best_case_hmean_with_score(pred_score, max_rule_evidence),
            )
            if not rule_can_survive(fact_kb1[alignment_base.OBJ], subj2_id, upper_bound_score):
                profile_stats['upper_bound_pruned_evidence'] += 1
                continue
            add_entity_evidence_pair(context, subj2_id, single_evi2, fact_kb1, score)

    def context_pair_count(context):
        """Count the total number of source-target entity pairs in the context."""
        return sum(len(pair_map) for pair_map in context['subj2_pairs'].values())

    def align_context(context):
        nonlocal pending_pairs
        subj2_pairs = context['subj2_pairs']
        entity_memory_profile = context['entity_memory_profile']

        # Select the target entity candidates with the most evidence pairs, skipping any that already have a strong alignment score.
        stage_start = time.perf_counter()
        subj2_count = dict()
        maxCount = 0
        for subj2, pair_map in subj2_pairs.items():
            if has_ent_score_dst(subj2) and round(max_ent_score_dst(subj2), 1) >= 1.0:
                continue
            cur_count = len(pair_map)
            if cur_count > maxCount: # keep only the candidates with the most evidence pairs
                subj2_count = dict()
                maxCount = cur_count
                subj2_count[subj2] = cur_count
            elif cur_count == maxCount:
                subj2_count[subj2] = cur_count
        select_time = time.perf_counter() - stage_start
        profile_stats['select'] += select_time
        context['timings']['select'] += select_time
        context['selected_candidates'] = len(subj2_count)

        # Align the source entity with each selected target entity candidate using the collected evidence pairs.
        stage_start = time.perf_counter()
        gramN = min(20, maxCount)
        entity_memory_profile['maxCount'] = maxCount
        for subj_kb2_id in subj2_count:
            sorted_pairs = sorted(subj2_pairs[subj_kb2_id].items(), reverse=True, key=lambda item: item[1][1])
            if len(sorted_pairs) > entity_memory_profile['sorted_pairs_peak']:
                entity_memory_profile['sorted_pairs_peak'] = len(sorted_pairs)
            ev2s = [pair[0] for pair in sorted_pairs]
            ev1s = [pair[1][0] for pair in sorted_pairs]

            visited_facts = set()
            for n in range(1, gramN + 1):
                ev1, ev2 = ev1s[:n], ev2s[:n]
                if (tuple(ev1), tuple(ev2)) in visited_facts:
                    continue
                visited_facts.add((tuple(ev1), tuple(ev2)))
                if len(visited_facts) > entity_memory_profile['visited_facts_peak']:
                    entity_memory_profile['visited_facts_peak'] = len(visited_facts)
                obj1_combo, pred1_combo, subj1_combo = zip(*ev1)
                obj2_combo, pred2_combo, subj2_combo = zip(*ev2)
                assert len(set(subj1_combo)) == 1
                assert len(set(subj2_combo)) == 1
                if not _have_same_repetition_pattern(obj1_combo, obj2_combo):
                    continue
                localfunc1 = kb_src.localFunctionalityIds(obj1_combo, pred1_combo)
                localfunc2 = kb_dst.localFunctionalityIds(obj2_combo, pred2_combo)
                pred1_sort = tuple(sorted(pred1_combo))
                pred2_sort = tuple(sorted(pred2_combo))
                globalfunc1 = compact_group_functionality(
                    src_group_functionalities,
                    src_functionalities,
                    pred1_sort,
                )
                globalfunc2 = compact_group_functionality(
                    dst_group_functionalities,
                    dst_functionalities,
                    pred2_sort,
                )

                obj_eq = alignment_base.fast_hmean(
                    active_pair_score(obj1_combo[i], obj2_combo[i])
                    for i in range(len(obj1_combo))
                )
                pred_eq = alignment_base.fast_hmean(
                    quasi_scores[pred1_combo[i]][pred2_combo[i]]
                    for i in range(len(pred1_combo))
                )

                if n == 1:
                    body_values = (
                        obj_eq, pred_eq, localfunc1, localfunc2,
                        src_functionalities[pred1_combo[0]],
                        dst_functionalities[pred2_combo[0]],
                    )
                else:
                    body_values = (
                        obj_eq, pred_eq, localfunc1, localfunc2,
                        globalfunc1, globalfunc2,
                    )
                rule_score = min(min(body_values), 1.0)
                if not rule_can_survive(subj1_combo[0], subj2_combo[0], rule_score):
                    profile_stats['upper_bound_pruned_rules'] += 1
                    continue
                created_pair = alignment_base.updateScoreMin(
                    ent_match_scores, subj1_combo[0], subj2_combo[0],
                    *body_values,
                )
                if created_pair:
                    pending_pairs += 1
                if pending_pairs >= alignment_base.MATCH_RESULT_FLUSH_PAIRS:
                    flush_compact_match_scores()
        align_time = time.perf_counter() - stage_start
        profile_stats['align'] += align_time
        context['timings']['align'] += align_time

    def finish_context(context):
        log._record_entity_expansion_profile(
            profile_stats,
            context['timings'],
            context['entity_memory_profile'],
            context_pairs=context.get('context_pairs', 0),
        )

    while True:
        entity_chunk = queue.get()
        if entity_chunk is None:
            break

        for subj_kb1 in entity_chunk:
            profile_stats['entities_seen'] += 1
            subj_kb1_id = side_keys.coerce_entity_id(kb_src, subj_kb1)
            if subj_kb1_id is None:
                continue

            if kb_src.is_literal_id(subj_kb1_id):
                continue

            if has_ent_score_src(subj_kb1_id) and round(max_ent_score_src(subj_kb1_id), 1) >= 1.0:
                profile_stats['skipped_matched'] += 1
                continue
            profile_stats['entities_processed'] += 1
            context = {
                'subj_kb1_id': subj_kb1_id,
                'subj2_pairs': {},
                'timings': {
                    'build_facts': 0.0,
                    'collect_pairs': 0.0,
                    'select': 0.0,
                    'align': 0.0,
                },
                'facts_kept': 0,
                'context_pairs': 0,
                'selected_candidates': 0,
                'entity_memory_profile': {
                    'candidate_obj2': 0,
                    'scanned_evi2': 0,
                    'aligned_evi2_peak': 0,
                    'tmp_subj2_peak': 0,
                    'maxCount': 0,
                    'sorted_pairs_peak': 0,
                    'visited_facts_peak': 0,
                },
            }

            stage_start = time.perf_counter()
            kb1_facts_ordered = []
            for _, predicate_id, object_id in kb_src.triplesWithSubjectIds(subj_kb1_id):
                obj_match_score = max_ent_score_src(object_id)
                inverse_predicate_id = kb_src._inverse_predicate_id(predicate_id)
                if inverse_predicate_id is None:
                    continue
                pred_match_score = max_pred_score(predicate_id)
                if obj_match_score <= 0:
                    continue
                if pred_match_score <= 0:
                    continue
                kb1_facts_ordered.append((object_id, inverse_predicate_id, subj_kb1_id))
            kb1_facts_ordered.sort(
                reverse=True,
                key=lambda x: min(max_ent_score_src(x[alignment_base.SUBJ]), max_pred_score(x[alignment_base.PRED])),
            )
            build_time = time.perf_counter() - stage_start
            profile_stats['build_facts'] += build_time
            context['timings']['build_facts'] += build_time
            context['facts_kept'] = len(kb1_facts_ordered)

            stage_start = time.perf_counter()
            candidate_facts = kb1_facts_ordered[:params['gramN']]
            for fact_kb1 in candidate_facts:
                pred_kb1, obj_kb1 = fact_kb1[alignment_base.PRED], fact_kb1[alignment_base.SUBJ]
                quasi_score_map = quasi_scores.get(pred_kb1)
                if quasi_score_map:
                    quasi_predicate_ids = quasi_predicates[pred_kb1]
                    obj_kb2_scores = tuple(active_target_scores(obj_kb1))
                    if obj_kb2_scores:
                        if not fact_can_survive_upper_bound(
                            fact_kb1,
                            quasi_score_map,
                            obj_kb2_scores,
                        ):
                            profile_stats['fact_upper_bound_pruned_facts'] += 1
                            continue
                        collect_fact_pairs_by_adjacency(
                            context,
                            fact_kb1,
                            quasi_score_map,
                            quasi_predicate_ids,
                            obj_kb2_scores,
                        )
            collect_time = time.perf_counter() - stage_start
            profile_stats['collect_pairs'] += collect_time
            context['timings']['collect_pairs'] += collect_time
            context_pairs = context_pair_count(context)
            context['context_pairs'] = context_pairs
            align_context(context)
            finish_context(context)
            del context

    flush_compact_match_scores()
    if profile_queue is not None:
        profile_queue.put(profile_stats)
    exit(0)

def _match_entities_by_rules(kb_src, kb_dst, quasiEqvirel, queue, ent_match_tuple_queue, ent_max_assign, functionalities, params, profile_queue=None):
    """ 
    Match entities in parallel using the rules, corresponding to equation (1) in the paper.
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
    ent_max_assign : dict
        The bilateral max assignment of entities
    functionalities : dict
        A dictionary mapping predicates to their functionalities
    params : dict
        A dictionary of parameters, including 'gramN' (the maximum n-gram size to consider)
    profile_queue : mp.Queue | None, optional
        Optional queue receiving worker timing and pruning statistics.

    Returns
    -------
    None
        Match scores and optional profile data are emitted through queues.
    """
    # Check if we can use the compact graph for efficient matching
    if alignment_base._can_use_compact_id_match(kb_src, kb_dst):
        return _match_entities_by_rules_compact(
            kb_src,
            kb_dst,
            quasiEqvirel,
            queue,
            ent_match_tuple_queue,
            ent_max_assign,
            functionalities,
            params,
            profile_queue=profile_queue,
        )

    ent_match_scores = dict()
    pending_pairs = 0
    profile_stats = {
        'build_facts': 0.0, # seconds spent finding and ranking usable source facts
        'collect_pairs': 0.0, # seconds spent collecting target evidence pairs
        'select': 0.0, # seconds spent selecting target entity candidates
        'align': 0.0, # seconds spent applying entity alignment rules
        'entities_seen': 0, # source entities read from the task queue
        'entities_processed': 0, # non-literal, non-finalized source entities processed
        'skipped_matched': 0, # source entities skipped because they already match strongly
        'fact_upper_bound_pruned_facts': 0, # source facts skipped by best-case score bound
        'target_score_upper_bound_pruned_predicates': 0, # target predicates skipped by score bound
        'target_score_upper_bound_pruned_candidates': 0, # target object candidates losing all predicates to score bound
        'upper_bound_pruned_evidence': 0, # evidence pairs skipped by best-case score bound
        'upper_bound_pruned_rules': 0, # final rule applications skipped by best-case score bound
        'target_hub_functionality_pruned_predicates': 0, # target predicates skipped for high degree
        'target_hub_functionality_pruned_candidates': 0, # target object candidates losing all predicates to high degree
        'total_candidate_obj2': 0, # target object candidates considered across entities
        'total_scanned_evi2': 0, # target evidence triples scanned across entities
        'max_scanned_evi2_entity': 0, # largest scanned target evidence count for one entity
        'max_context_pairs_entity': 0, # largest retained evidence-pair count for one entity
        'max_align_time_entity_s': 0.0, # longest rule-application time for one entity
    }
    # quasi predicate scores
    positive_quasi_scores = {
        predicate: {
            target_predicate: score
            for target_predicate, score in target_scores.items()
            if score > 0
        }
        for predicate, target_scores in quasiEqvirel.items()
    }
    max_pred_score_cache = {
        predicate: max(target_scores.values())
        for predicate, target_scores in positive_quasi_scores.items()
        if target_scores
    }
    minimum_output_score = max(0.0, float(params.get('prune_min_score', 0.0) or 0.0))
    target_hub_degree_threshold = params.get('target_hub_degree_threshold', 10000)
    target_hub_degree_threshold = (
        target_hub_degree_threshold
        if target_hub_degree_threshold and target_hub_degree_threshold > 0
        else None
    )
    def group_functionality(predicates):
        return functionalities.get(predicates, 1.0)

    max_ent_score_cache = OrderedDict()

    # LRU cache for max entity scores
    def lru_get(cache, key):
        try:
            value = cache.pop(key)
        except KeyError:
            return None
        cache[key] = value
        return value

    def lru_put(cache, key, value, maxsize):
        cache[key] = value
        if len(cache) > maxsize:
            cache.popitem(last=False)

    def max_ent_score(entity):
        cached_score = lru_get(max_ent_score_cache, entity)
        if cached_score is not None:
            return cached_score
        score = max(ent_max_assign.get(entity, {None: 0}).values())
        lru_put(max_ent_score_cache, entity, score, alignment_base.MAX_ENT_SCORE_CACHE_SIZE)
        return score

    def max_pred_score(predicate):
        return max_pred_score_cache.get(predicate, 0)

    def current_pair_score(subj1, subj2):
        """Get the current best score of a source-target entity pair."""
        # alignment scores generated by this worker in current iteration
        generated_scores = ent_match_scores.get(subj1)
        generated_score = 0.0 if generated_scores is None else generated_scores.get(subj2, 0.0)
        # alignment scores from the bilateral max assignment
        active_scores = ent_max_assign.get(subj1)
        active_score = 0.0 if active_scores is None else active_scores.get(subj2, 0.0)
        return max(generated_score, active_score)

    def rule_score_floor(subj1, subj2):
        """Get the minimum score floor for a source-target entity pair based on current scores and max entity scores."""
        return max(
            minimum_output_score,
            current_pair_score(subj1, subj2),
            max_ent_score(subj1),
            max_ent_score(subj2),
        )

    def rule_can_survive(subj1, subj2, upper_bound):
        """Check if a score can survive the upper bound score pruning."""
        return upper_bound > rule_score_floor(subj1, subj2)

    def best_case_hmean_with_score(score, max_count):
        """Compute the best-case harmonic mean of a score with a maximum count of evidence."""
        if score <= 0:
            return 0.0
        if score >= 1.0:
            return 1.0
        max_count = max(1, int(max_count))
        return max_count / ((1.0 / score) + max_count - 1) # all other evidence is perfect (score=1.0)

    def fact_can_survive_upper_bound(fact_kb1, quasi_score_map, obj_kb2_scores):
        """Check if a source fact can survive the upper bound score pruning considering the best object and predicate scores."""
        if not quasi_score_map or not obj_kb2_scores:
            return False

        best_obj_score = max((score for _, score in obj_kb2_scores), default=0.0)
        best_pred_score = max(quasi_score_map.values(), default=0.0)
        if best_obj_score <= 0 or best_pred_score <= 0:
            return False
        max_rule_evidence = min(20, max(1, params.get('gramN', 20)))
        fact_upper_bound = min(
            best_case_hmean_with_score(best_obj_score, max_rule_evidence),
            best_case_hmean_with_score(best_pred_score, max_rule_evidence),
        )
        return fact_upper_bound > max(minimum_output_score, max_ent_score(fact_kb1[alignment_base.OBJ]))

    def build_predicate_bound_filter(fact_kb1, quasi_scores):
        """Precompute predicate-side score bounds once for a source fact."""
        floor = max(minimum_output_score, max_ent_score(fact_kb1[alignment_base.OBJ]))
        max_rule_evidence = min(20, max(1, params.get('gramN', 20)))
        always_pruned = 0
        surviving_predicates = []
        for predicate, pred_score in quasi_scores.items():
            pred_bound = best_case_hmean_with_score(pred_score, max_rule_evidence)
            if pred_bound <= floor:
                always_pruned += 1
            else:
                surviving_predicates.append(predicate)
        return floor, max_rule_evidence, always_pruned, tuple(surviving_predicates)

    def target_predicate_object_count(subject, predicate):
        """Get the number of objects for a given subject and predicate in the target knowledge base."""
        objects = kb_dst._objects_for_subject_predicate(subject, predicate)
        return len(objects) if objects else 0

    def filter_target_predicates_by_functionality_bound(
            fact_kb1,
            obj_kb2,
            obj_score,
            quasi_scores,
            predicate_bound_filter=None):
        """Filter target predicates based on functionality and score upper bounds."""
        predicates = tuple(quasi_scores)
        hub_pruning_enabled = target_hub_degree_threshold is not None
        if predicate_bound_filter is None:
            predicate_bound_filter = build_predicate_bound_filter(fact_kb1, quasi_scores)
        floor, max_rule_evidence, score_pruned_predicates, predicates = predicate_bound_filter
        obj_bound = best_case_hmean_with_score(obj_score, max_rule_evidence)
        if obj_bound <= floor:
            profile_stats['target_score_upper_bound_pruned_predicates'] += (
                score_pruned_predicates + len(predicates)
            )
            profile_stats['target_score_upper_bound_pruned_candidates'] += 1
            return ()
        if not predicates:
            if score_pruned_predicates:
                profile_stats['target_score_upper_bound_pruned_predicates'] += score_pruned_predicates
                profile_stats['target_score_upper_bound_pruned_candidates'] += 1
            return ()

        # Hub pruning is the only reason to query per subject-predicate counts;
        # nonexistent predicates naturally yield no evidence during scanning.
        if not hub_pruning_enabled:
            if score_pruned_predicates:
                profile_stats['target_score_upper_bound_pruned_predicates'] += score_pruned_predicates
            return predicates

        kept_predicates = []
        hub_pruned_predicates = 0

        for predicate in predicates:
            object_count = target_predicate_object_count(obj_kb2, predicate)
            if object_count <= 0:
                continue

            if hub_pruning_enabled and object_count >= target_hub_degree_threshold:
                hub_pruned_predicates += 1
                continue

            kept_predicates.append(predicate)

        if score_pruned_predicates:
            profile_stats['target_score_upper_bound_pruned_predicates'] += score_pruned_predicates
        if hub_pruned_predicates:
            profile_stats['target_hub_functionality_pruned_predicates'] += hub_pruned_predicates
        if not kept_predicates:
            if score_pruned_predicates:
                profile_stats['target_score_upper_bound_pruned_candidates'] += 1
            if hub_pruned_predicates:
                profile_stats['target_hub_functionality_pruned_candidates'] += 1
        return tuple(kept_predicates)

    def add_entity_evidence_pair(subj2_pairs, subj2, evi2, fact_kb1, score):
        """Add a source-target entity pair with evidence to the subj2_pairs, keeping only the best score for each target entity."""
        pair_map = subj2_pairs.get(subj2)
        if pair_map is None:
            pair_map = {}
            subj2_pairs[subj2] = pair_map
        previous_pair = pair_map.get(evi2)
        if previous_pair is None or score > previous_pair[1]:
            pair_map[evi2] = (fact_kb1, score)

    def collect_fact_pairs_by_adjacency(
            subj2_pairs,
            fact_kb1,
            quasi_score_map,
            obj_kb2_scores,
            entity_memory_profile):
        """Collect target entity candidates and their evidence for a given source fact.
        Parameters:
            subj2_pairs: A dictionary mapping target entities to their evidence pairs.
            fact_kb1: The source fact.
            quasi_score_map: A map of quasi-scores for predicates.
            obj_kb2_scores: A list of target object scores.
            entity_memory_profile: A dictionary to track memory usage statistics for the current entity."""
        tmp_subj2_evi2 = {}
        subj2_maxsubrel_score = {}
        predicate_bound_filter = build_predicate_bound_filter(fact_kb1, quasi_score_map)

        for obj_kb2, obj_score in sorted(
            obj_kb2_scores,
            key=lambda item: (-item[1], item[0]),
        ):
            entity_memory_profile['candidate_obj2'] += 1
            if not kb_dst.has_subject(obj_kb2):
                continue

            filtered_quasi_predicates = filter_target_predicates_by_functionality_bound(fact_kb1, obj_kb2, obj_score, quasi_score_map, predicate_bound_filter)
            if not filtered_quasi_predicates:
                continue

            aligned_evi2 = []
            maxsubrel_score = 0
            for evi2_ in kb_dst.triplesWithSubject(obj_kb2, filtered_quasi_predicates):
                entity_memory_profile['scanned_evi2'] += 1
                if utils.isLiteral(evi2_[alignment_base.OBJ]):
                    continue
                subrel_score = quasi_score_map[evi2_[alignment_base.PRED]]
                if subrel_score > maxsubrel_score:
                    maxsubrel_score = subrel_score
                    aligned_evi2 = [evi2_]
                elif subrel_score == maxsubrel_score:
                    aligned_evi2.append(evi2_)

            if len(aligned_evi2) > entity_memory_profile['aligned_evi2_peak']:
                entity_memory_profile['aligned_evi2_peak'] = len(aligned_evi2)

            for evi2 in aligned_evi2:
                subj2 = evi2[alignment_base.OBJ]
                subrel_score = quasi_score_map[evi2[alignment_base.PRED]]
                if subrel_score > subj2_maxsubrel_score.get(subj2, 0):
                    subj2_maxsubrel_score[subj2] = subrel_score
                    tmp_subj2_evi2[subj2] = evi2

        if len(tmp_subj2_evi2) > entity_memory_profile['tmp_subj2_peak']:
            entity_memory_profile['tmp_subj2_peak'] = len(tmp_subj2_evi2)

        for subj2, single_evi2 in tmp_subj2_evi2.items():
            obj_score = ent_max_assign[fact_kb1[alignment_base.SUBJ]][single_evi2[alignment_base.SUBJ]]
            pred_score = positive_quasi_scores[fact_kb1[alignment_base.PRED]][single_evi2[alignment_base.PRED]]
            score = min(obj_score, pred_score)
            max_rule_evidence = min(20, max(1, params.get('gramN', 20)))
            upper_bound_score = min(
                best_case_hmean_with_score(obj_score, max_rule_evidence),
                best_case_hmean_with_score(pred_score, max_rule_evidence),
            )
            if not rule_can_survive(fact_kb1[alignment_base.OBJ], subj2, upper_bound_score):
                profile_stats['upper_bound_pruned_evidence'] += 1
                continue
            add_entity_evidence_pair(subj2_pairs, subj2, single_evi2, fact_kb1, score)

    while True:
        entity_chunk = queue.get()
        if entity_chunk is None:
            break

        for subj_kb1 in entity_chunk:
            profile_stats['entities_seen'] += 1

            # We don't need to match literals
            if utils.isLiteral(subj_kb1):
                continue

            # Skip if the entity is already matched
            if subj_kb1 in ent_max_assign and round(max_ent_score(subj_kb1), 1) >= 1.0:
                profile_stats['skipped_matched'] += 1
                continue
            profile_stats['entities_processed'] += 1
            entity_memory_profile = {
                'candidate_obj2': 0,
                'scanned_evi2': 0,
                'aligned_evi2_peak': 0,
                'tmp_subj2_peak': 0,
                'maxCount': 0,
                'sorted_pairs_peak': 0,
                'visited_facts_peak': 0,
            }
            entity_timings = {'build_facts': 0.0, 'collect_pairs': 0.0, 'select': 0.0, 'align': 0.0}

            # Candidate search: rank usable source facts by current alignment strength.
            stage_start = time.perf_counter()
            kb1_facts_ordered = []
            for fact1 in kb_src.triplesWithSubject(subj_kb1):
                obj_match_score = max_ent_score(fact1[alignment_base.OBJ])
                pred_match_score = max_pred_score(fact1[alignment_base.PRED])
                if obj_match_score <= 0:
                    continue
                if pred_match_score <= 0:
                    continue
                kb1_facts_ordered.append((fact1[alignment_base.OBJ], utils.invert(fact1[alignment_base.PRED]), subj_kb1))
            kb1_facts_ordered.sort(
                reverse=True,
                key=lambda x: min(max_ent_score(x[alignment_base.SUBJ]), max_pred_score(x[alignment_base.PRED])),
            )
            build_time = time.perf_counter() - stage_start
            profile_stats['build_facts'] += build_time
            entity_timings['build_facts'] += build_time

            # {subj2: {evi2: (best_evi1, pair_score)}}.
            # Using dictionaries avoids repeated linear scans over evidence lists.
            stage_start = time.perf_counter()
            subj2_pairs = dict()
            candidate_facts = kb1_facts_ordered[:params['gramN']]
            for fact_kb1 in candidate_facts:
                pred_kb1, obj_kb1 = fact_kb1[alignment_base.PRED], fact_kb1[alignment_base.SUBJ]
                quasi_score_map = positive_quasi_scores.get(pred_kb1)
                if not quasi_score_map:
                    continue
                obj_kb2_scores = tuple(ent_max_assign[obj_kb1].items())
                if not fact_can_survive_upper_bound(
                    fact_kb1,
                    quasi_score_map,
                    obj_kb2_scores,
                ):
                    profile_stats['fact_upper_bound_pruned_facts'] += 1
                    continue
                collect_fact_pairs_by_adjacency(
                    subj2_pairs,
                    fact_kb1,
                    quasi_score_map,
                    obj_kb2_scores,
                    entity_memory_profile,
                )
            collect_time = time.perf_counter() - stage_start
            profile_stats['collect_pairs'] += collect_time
            entity_timings['collect_pairs'] += collect_time
            context_pairs = sum(len(pair_map) for pair_map in subj2_pairs.values())

            # Selection Algorithm
            # select the entities with the most evidences
            stage_start = time.perf_counter()
            subj2_count = dict()
            maxCount = 0
            for subj2, pair_map in subj2_pairs.items():
                if subj2 in ent_max_assign and round(max_ent_score(subj2), 1) >= 1.0:
                    continue
                cur_count = len(pair_map)
                if cur_count > maxCount:
                    subj2_count = dict()
                    maxCount = cur_count
                    subj2_count[subj2] = cur_count
                elif cur_count == maxCount:
                    subj2_count[subj2] = cur_count
            select_time = time.perf_counter() - stage_start
            profile_stats['select'] += select_time
            entity_timings['select'] += select_time

            # Alignment Algorithm
            # Apply rules in order to update the scores
            stage_start = time.perf_counter()
            gramN = min(20, maxCount)
            entity_memory_profile['maxCount'] = maxCount
            for subj_kb2 in subj2_count:
                sorted_pairs = sorted(
                    subj2_pairs[subj_kb2].items(),
                    reverse=True,
                    key=lambda item: item[1][1],
                )
                if len(sorted_pairs) > entity_memory_profile['sorted_pairs_peak']:
                    entity_memory_profile['sorted_pairs_peak'] = len(sorted_pairs)
                ev2s = [pair[0] for pair in sorted_pairs]
                ev1s = [pair[1][0] for pair in sorted_pairs]

                # find the common patterns
                visited_facts = set()
                # Try all possible sets
                for n in range(1, gramN+1):
                    ev1, ev2 = ev1s[:n], ev2s[:n]
                    if (tuple(ev1), tuple(ev2)) in visited_facts:
                            continue
                    visited_facts.add((tuple(ev1), tuple(ev2)))
                    if len(visited_facts) > entity_memory_profile['visited_facts_peak']:
                        entity_memory_profile['visited_facts_peak'] = len(visited_facts)
                    obj1_combo, pred1_combo, subj1_combo = zip(*ev1)
                    obj2_combo, pred2_combo, subj2_combo = zip(*ev2)
                    # check if subjects itself are the same
                    assert len(set(subj1_combo)) == 1
                    assert len(set(subj2_combo)) == 1
                    # check same pattern
                    if not _have_same_repetition_pattern(obj1_combo, obj2_combo):
                        continue
                    localfunc1 = kb_src.localFunctionality(obj1_combo, pred1_combo)
                    localfunc2 = kb_dst.localFunctionality(obj2_combo, pred2_combo)
                    pred1_sort = tuple(sorted(list(pred1_combo)))
                    pred2_sort = tuple(sorted(list(pred2_combo)))
                    globalfunc1 = group_functionality(pred1_sort)
                    globalfunc2 = group_functionality(pred2_sort)

                    obj_eq = alignment_base.fast_hmean(
                        ent_max_assign[obj1_combo[i]][obj2_combo[i]]
                        for i in range(len(obj1_combo))
                    )
                    pred_eq = alignment_base.fast_hmean(
                        quasiEqvirel[pred1_combo[i]][pred2_combo[i]]
                        for i in range(len(pred1_combo))
                    )
                    # update
                    if n == 1:
                        body_values = (
                            obj_eq, pred_eq, localfunc1, localfunc2,
                            functionalities[pred1_combo[0]],
                            functionalities[pred2_combo[0]],
                        )
                    else:
                        body_values = (
                            obj_eq, pred_eq, localfunc1, localfunc2,
                            globalfunc1, globalfunc2,
                        )
                    rule_score = min(min(body_values), 1.0)
                    if not rule_can_survive(subj1_combo[0], subj2_combo[0], rule_score):
                        profile_stats['upper_bound_pruned_rules'] += 1
                        continue
                    created_pair = alignment_base.updateScoreMin(
                        ent_match_scores, subj1_combo[0], subj2_combo[0],
                        *body_values,
                    )
                    if created_pair:
                        pending_pairs += 1
                    if pending_pairs >= alignment_base.MATCH_RESULT_FLUSH_PAIRS:
                        ent_match_scores, pending_pairs = alignment_base._flush_match_scores(
                            ent_match_tuple_queue,
                            ent_match_scores,
                        )
            align_time = time.perf_counter() - stage_start
            profile_stats['align'] += align_time
            entity_timings['align'] += align_time
            log._record_entity_expansion_profile(
                profile_stats,
                entity_timings,
                entity_memory_profile,
                context_pairs=context_pairs,
            )
        ent_match_scores, pending_pairs = alignment_base._flush_match_scores(
            ent_match_tuple_queue,
            ent_match_scores,
        )
    ent_match_scores, pending_pairs = alignment_base._flush_match_scores(ent_match_tuple_queue, ent_match_scores)
    if profile_queue is not None:
        profile_queue.put(profile_stats)
    exit(0)