"""
This file is part of FLORA, an unsupervised system for automatic knowledge graph (KG) alignment. 
The file is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0) by Yiwen Peng, Thomas Bonald, Fabian Suchanek and Lingyun Huang.

Description: side key management utilities for keeping local compact graph IDs distinct across knowledge bases.
"""

import logging
import utils

# Entity sides.  Example entity keys: ('kb1', 12), ('kb2', 12).
KB1 = 'kb1'
KB2 = 'kb2'

# Predicate sides.  Predicate IDs have their own namespace in each graph.
PRED1 = 'p1'
PRED2 = 'p2'


def side_entity_key(side, entity_id):
    """Return a side-qualified entity key for a local integer entity ID."""
    return (side, int(entity_id))


def side_predicate_key(side, predicate_id):
    """Return a side-qualified predicate key for a local integer predicate ID."""
    return (side, int(predicate_id))


def is_side_key(value, side=None):
    """Return True when value is a side-qualified entity or predicate ID."""
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and value[0] in (KB1, KB2, PRED1, PRED2)
        and (side is None or value[0] == side)
        and isinstance(value[1], int)
    )


def is_id_keyed_mapping(mapping):
    """Return True when a nested score mapping already uses side keys."""
    for source, targets in mapping.items():
        if not is_side_key(source):
            return False
        for target in targets:
            return is_side_key(target)
        return True
    return False


def tag_graph_sides(kb1, kb2):
    """Attach side tags to graphs so later helpers can infer key namespaces."""
    kb1._flora_side = KB1
    kb2._flora_side = KB2


def graph_side(graph, default):
    """Return the entity side tag attached to a graph, or a caller default."""
    return getattr(graph, '_flora_side', default)


def graph_predicate_side(graph, default):
    """Return the predicate side tag corresponding to a graph's entity side."""
    side = graph_side(graph, default)
    if side in (PRED1, PRED2):
        return side
    return PRED1 if side == KB1 else PRED2


def coerce_entity_id(graph, entity):
    """Return the local integer ID from an int, side key, or graph entity."""
    if isinstance(entity, int):
        return entity
    if is_side_key(entity):
        return entity[1]
    return graph.entity_id(entity)


def decode_entity_key(entity, kb1=None, kb2=None):
    """Decode a side-qualified entity key back to the graph's entity value."""
    if not is_side_key(entity):
        return entity
    side, entity_id = entity
    graph = kb1 if side == KB1 else kb2 if side == KB2 else None
    if graph is None or not hasattr(graph, 'entity_for_id'):
        return entity
    return graph.entity_for_id(entity_id)


def graph_has_entity(graph, entity):
    """Return True when a graph appears to contain an entity or side-keyed ID."""
    if graph is None:
        return False
    if is_side_key(entity):
        return hasattr(graph, 'has_subject_id') and graph.has_subject_id(entity[1])
    if hasattr(graph, 'entity_id'):
        try:
            return graph.entity_id(entity) is not None
        except Exception:
            return False
    return hasattr(graph, 'has_subject') and graph.has_subject(entity)


def entity_side_label(entity, kb1=None, kb2=None):
    """Return a readable side label for diagnostics."""
    if is_side_key(entity, KB1):
        return 'kb1'
    if is_side_key(entity, KB2):
        return 'kb2'
    in_kb1 = graph_has_entity(kb1, entity)
    in_kb2 = graph_has_entity(kb2, entity)
    if in_kb1 and in_kb2:
        return 'both'
    if in_kb1:
        return 'kb1'
    if in_kb2:
        return 'kb2'
    if utils.isLiteral(entity):
        return 'literal'
    return 'unknown'


def decode_predicate_key(predicate, kb1=None, kb2=None):
    """Decode a side-qualified predicate key back to the graph's predicate value."""
    if not is_side_key(predicate):
        return predicate
    side, predicate_id = predicate
    graph = kb1 if side == PRED1 else kb2 if side == PRED2 else None
    if graph is None or not hasattr(graph, 'predicate_for_id'):
        return predicate
    return graph.predicate_for_id(predicate_id)


def can_use_id_keyed_state(kb1, kb2):
    """Return True when both graphs expose the ID APIs needed for side keys."""
    return (
        hasattr(kb1, 'entity_id')
        and hasattr(kb2, 'entity_id')
        and hasattr(kb1, 'predicate_id')
        and hasattr(kb2, 'predicate_id')
        and hasattr(kb1, 'entity_for_id')
        and hasattr(kb2, 'entity_for_id')
    )


def encode_entity_for_side(entity, preferred_graph, preferred_side, other_graph, other_side):
    """Encode an entity by trying the expected graph first, then the other graph."""
    entity_id = preferred_graph.entity_id(entity)
    if entity_id is not None:
        return side_entity_key(preferred_side, entity_id)
    entity_id = other_graph.entity_id(entity)
    if entity_id is not None:
        return side_entity_key(other_side, entity_id)
    return None


def maybe_encode_same_as_scores(scores, kb1, kb2):
    """
    Convert a string-keyed sameAs score mapping to side-keyed IDs when possible.

    Example
    -------
    If kb1.entity_id('a') == 1 and kb2.entity_id('b') == 2:

    {'a': {'b': 0.95}} becomes {('kb1', 1): {('kb2', 2): 0.95}}.
    """
    if not can_use_id_keyed_state(kb1, kb2) or is_id_keyed_mapping(scores):
        return scores
    encoded = {}
    for entity1, target_scores in scores.items():
        entity1_key = encode_entity_for_side(entity1, kb1, KB1, kb2, KB2)
        if entity1_key is None:
            continue
        source_side = entity1_key[0]
        preferred_graph = kb2 if source_side == KB1 else kb1
        preferred_side = KB2 if source_side == KB1 else KB1
        other_graph = kb1 if source_side == KB1 else kb2
        other_side = KB1 if source_side == KB1 else KB2
        for entity2, score in target_scores.items():
            entity2_key = encode_entity_for_side(entity2, preferred_graph, preferred_side, other_graph, other_side)
            if entity2_key is None:
                continue
            encoded.setdefault(entity1_key, {})
            if score > encoded[entity1_key].get(entity2_key, 0):
                encoded[entity1_key][entity2_key] = score
    logging.info(
        "Encoded sameAsScores to side/id keys | sources=%s | pairs=%s",
        len(encoded), sum(len(values) for values in encoded.values()),
    )
    return encoded


def predicate_side_keys(predicate, kb1, kb2):
    """
    Return all side-qualified predicate keys matching a predicate string.
    (The same predicate string can exist in both graphs with different local IDs.)
    """
    keys = []
    predicate1_id = kb1.predicate_id(predicate)
    if predicate1_id is not None:
        keys.append(side_predicate_key(PRED1, predicate1_id))
    predicate2_id = kb2.predicate_id(predicate)
    if predicate2_id is not None:
        keys.append(side_predicate_key(PRED2, predicate2_id))
    return keys


def maybe_encode_predicate_mapping(mapping, kb1, kb2, preserve_mapping=None):
    """
    Convert a string-keyed predicate score mapping to side-keyed IDs.

    Example
    -------
    Suppose "rdf:type" is ID 3 in kb1 and ID 7 in kb2:
    {'rdf:type': {'rdf:type': 1.0}} becomes:
    {('p1', 3): {('p2', 7): 1.0}, ('p2', 7): {('p1', 3): 1.0}}.
    """
    if ((preserve_mapping is not None and preserve_mapping(mapping))
        or not can_use_id_keyed_state(kb1, kb2)
        or is_id_keyed_mapping(mapping)):
        return mapping
    encoded = {}
    for predicate1, target_scores in mapping.items():
        predicate1_keys = predicate_side_keys(predicate1, kb1, kb2)
        if not predicate1_keys:
            continue
        for predicate2, score in target_scores.items():
            predicate2_keys = predicate_side_keys(predicate2, kb1, kb2)
            if not predicate2_keys:
                continue
            for predicate1_key in predicate1_keys:
                for predicate2_key in predicate2_keys:
                    if predicate1_key[0] == predicate2_key[0]:
                        continue
                    encoded.setdefault(predicate1_key, {})
                    if score > encoded[predicate1_key].get(predicate2_key, 0):
                        encoded[predicate1_key][predicate2_key] = score
    logging.info(
        "Encoded predicate mapping to side/id keys | predicates=%s | pairs=%s",
        len(encoded), sum(len(values) for values in encoded.values()),
    )
    return encoded


def maybe_encode_functionalities(functionalities, kb1, kb2):
    """
    Convert predicate functionality maps to side-keyed predicate IDs.

    Examples
    --------
    If kb1.predicate_id('p') == 4 and kb2.predicate_id('p') == 9:
    {'p': 0.8} becomes {('p1', 4): 0.8, ('p2', 9): 0.8}.

    Grouped predicate keys are encoded per graph when the whole group exists:
    {('p', 'q'): 0.5} may become
    {(('p1', 4), ('p1', 5)): 0.5, (('p2', 9), ('p2', 10)): 0.5}.
    """
    if not can_use_id_keyed_state(kb1, kb2):
        return functionalities
    encoded = {}
    for predicate_key, value in functionalities.items():
        if is_side_key(predicate_key):
            encoded[predicate_key] = value
            continue
        if isinstance(predicate_key, tuple):
            if predicate_key and all(is_side_key(predicate) for predicate in predicate_key):
                encoded[predicate_key] = value
                continue
            for graph, side in ((kb1, PRED1), (kb2, PRED2)):
                predicate_ids = []
                for predicate in predicate_key:
                    predicate_id = graph.predicate_id(predicate)
                    if predicate_id is None:
                        break
                    predicate_ids.append(side_predicate_key(side, predicate_id))
                else:
                    encoded[tuple(predicate_ids)] = value
            continue

        predicate1_id = kb1.predicate_id(predicate_key)
        if predicate1_id is not None:
            encoded[side_predicate_key(PRED1, predicate1_id)] = value
        predicate2_id = kb2.predicate_id(predicate_key)
        if predicate2_id is not None:
            encoded[side_predicate_key(PRED2, predicate2_id)] = value
    logging.info("Encoded functionalities to side/id keys | entries=%s", len(encoded))
    return encoded


def encode_compact_entity_assign(kb_src, kb_dst, ent_max_assign):
    """
    Convert entity assignments into local-ID maps for compact workers.

    Examples
    --------:
    {('kb1', 1): {('kb2', 2): 0.9}} becomes
    src_to_dst_scores == {1: {2: 0.9}}.
    """
    src_to_dst_scores = {}
    src_max_scores = {}
    dst_max_scores = {}

    # handle the normal side-keyed compact state without decoding IDs
    if is_id_keyed_mapping(ent_max_assign):
        src_side = graph_side(kb_src, KB1)
        dst_side = graph_side(kb_dst, KB2)
        for entity_key, target_scores in ent_max_assign.items():
            if not target_scores:
                continue
            side, entity_id = entity_key
            max_score = max(target_scores.values())
            if side == src_side:
                src_max_scores[entity_id] = max_score
                target_id_scores = {
                    target_key[1]: score
                    for target_key, score in target_scores.items()
                    if is_side_key(target_key, dst_side)
                }
                if target_id_scores:
                    src_to_dst_scores[entity_id] = target_id_scores
            elif side == dst_side:
                dst_max_scores[entity_id] = max_score
        return src_to_dst_scores, src_max_scores, dst_max_scores

    # support string-keyed maps by resolving graph-local IDs
    for entity, target_scores in ent_max_assign.items():
        if not target_scores:
            continue
        max_score = max(target_scores.values())

        src_id = kb_src.entity_id(entity)
        if src_id is not None:
            src_max_scores[src_id] = max_score
            target_id_scores = {}
            for target_entity, score in target_scores.items():
                target_id = kb_dst.entity_id(target_entity)
                if target_id is not None:
                    target_id_scores[target_id] = score
            if target_id_scores:
                src_to_dst_scores[src_id] = target_id_scores

        dst_id = kb_dst.entity_id(entity)
        if dst_id is not None:
            dst_max_scores[dst_id] = max_score

    return src_to_dst_scores, src_max_scores, dst_max_scores


def side_key_compact_predicate_mapping(predicate_mapping_ids, src_side, dst_side):
    """
    Wrap local-ID predicate mapping keys in source/target side tags.

    Compact subrelation workers return plain IDs for their current direction,
    for example {3: {7: 0.82}}.  Before merging with global state, those IDs
    must be tagged as {('p1', 3): {('p2', 7): 0.82}}.
    """
    return {
        side_predicate_key(src_side, predicate_id): {
            side_predicate_key(dst_side, target_predicate_id): score
            for target_predicate_id, score in target_scores.items()
        }
        for predicate_id, target_scores in predicate_mapping_ids.items()
    }


def encode_compact_predicate_scores(kb_src, kb_dst, predicate_scores):
    """
    Convert predicate score mappings into compact local-ID lookup tables.

    Example
    -------
    {('p1', 3): {('p2', 7): 0.8, ('p2', 9): 0.4}} returns:

    encoded_scores == {3: {7: 0.8, 9: 0.4}}
    encoded_predicates == {3: (7, 9)}
    max_scores == {3: 0.8}
    """
    encoded_scores = {}
    encoded_predicates = {}
    max_scores = {}

    # fast path for side-keyed predicate mappings.
    if is_id_keyed_mapping(predicate_scores):
        src_side = graph_predicate_side(kb_src, PRED1)
        dst_side = graph_predicate_side(kb_dst, PRED2)
        for predicate_key, target_scores in predicate_scores.items():
            if not is_side_key(predicate_key, src_side):
                continue
            predicate_id = predicate_key[1]
            target_id_scores = {
                target_key[1]: score
                for target_key, score in target_scores.items()
                if score > 0 and is_side_key(target_key, dst_side)
            }
            if not target_id_scores:
                continue
            encoded_scores[predicate_id] = target_id_scores
            encoded_predicates[predicate_id] = tuple(target_id_scores)
            max_scores[predicate_id] = max(target_id_scores.values())
        return encoded_scores, encoded_predicates, max_scores

    # fallback for string-keyed predicate maps.
    for predicate, target_scores in predicate_scores.items():
        predicate_id = kb_src.predicate_id(predicate)
        if predicate_id is None:
            continue
        target_id_scores = {}
        for target_predicate, score in target_scores.items():
            if score <= 0:
                continue
            target_predicate_id = kb_dst.predicate_id(target_predicate)
            if target_predicate_id is not None:
                target_id_scores[target_predicate_id] = score
        if not target_id_scores:
            continue
        encoded_scores[predicate_id] = target_id_scores
        encoded_predicates[predicate_id] = tuple(target_id_scores)
        max_scores[predicate_id] = max(target_id_scores.values())

    return encoded_scores, encoded_predicates, max_scores


def encode_compact_functionalities(graph, functionalities):
    """
    Split functionality values into single/grouped local predicate-ID maps.

    Example
    -------
    {('p1', 3): 0.8, (('p1', 3), ('p1', 5)): 0.4, ('p2', 7): 0.6}
    returns single == {3: 0.8} and grouped == {(3, 5): 0.4}.
    """
    single = {}
    grouped = {}
    predicate_side = graph_predicate_side(graph, PRED1)
    for predicate_key, value in functionalities.items():
        # fast path for already-local single predicate IDs.
        if isinstance(predicate_key, int):
            single[predicate_key] = value
            continue

        if isinstance(predicate_key, tuple):
            # distinguish side-keyed single predicates from grouped keys.
            if is_side_key(predicate_key):
                if is_side_key(predicate_key, predicate_side):
                    single[predicate_key[1]] = value
                continue
            if predicate_key and all(isinstance(predicate, int) for predicate in predicate_key):
                grouped[predicate_key] = value
                continue
            if predicate_key and all(is_side_key(predicate, predicate_side) for predicate in predicate_key):
                grouped[tuple(predicate[1] for predicate in predicate_key)] = value
                continue

            # resolve string or mixed grouped predicates against this graph.
            predicate_ids = []
            for predicate in predicate_key:
                if is_side_key(predicate):
                    if not is_side_key(predicate, predicate_side):
                        break
                    predicate_ids.append(predicate[1])
                    continue
                predicate_id = graph.predicate_id(predicate)
                if predicate_id is None:
                    break
                predicate_ids.append(predicate_id)
            else:
                grouped[tuple(predicate_ids)] = value
            continue

        # resolve string-keyed single predicates.
        predicate_id = graph.predicate_id(predicate_key)
        if predicate_id is not None:
            single[predicate_id] = value
    return single, grouped


def preencode_compact_worker_functionalities(src_functionalities, dst_functionalities):
    """
    Precompute compact worker functionality tables from already-local maps.

    Example
    -------
    src_functionalities == {3: 0.8, (3, 5): 0.4} becomes
    src_single == {3: 0.8} and src_grouped == {(3, 5): 0.4}.
    """
    def split_local(functionalities):
        single = {}
        grouped = {}
        for predicate_key, value in functionalities.items():
            if isinstance(predicate_key, int):
                single[predicate_key] = value
            elif isinstance(predicate_key, tuple):
                grouped[predicate_key] = value
        return single, grouped

    src_single, src_grouped = split_local(src_functionalities)
    dst_single, dst_grouped = split_local(dst_functionalities)
    return {
        '_compact_worker_functionalities': True,
        'src_single': src_single,
        'src_grouped': src_grouped,
        'dst_single': dst_single,
        'dst_grouped': dst_grouped,
    }


def resolve_compact_worker_functionalities(kb_src, kb_dst, functionalities):
    """Return compact functionality tables, unpacking or encoding as needed."""
    if (isinstance(functionalities, dict)
        and functionalities.get('_compact_worker_functionalities') is True):
        return (
            functionalities['src_single'],
            functionalities['src_grouped'],
            functionalities['dst_single'],
            functionalities['dst_grouped'],
        )
    src_single, src_grouped = encode_compact_functionalities(kb_src, functionalities)
    dst_single, dst_grouped = encode_compact_functionalities(kb_dst, functionalities)
    return src_single, src_grouped, dst_single, dst_grouped