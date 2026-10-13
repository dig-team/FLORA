"""
This file is part of FLORA, an unsupervised system for automatic knowledge graph (KG) alignment. 
The file is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0) by Yiwen Peng, Thomas Bonald, Fabian Suchanek and Lingyun Huang.

Description: Literal matching and literal-score initialization utilities, including exact and embedding-based matching.
This module can be run independently from FLORA's main loop. Because faiss-based literal matching benefits from GPU acceleration, 
literal matching results can be precomputed on a GPU-enabled machine, saved to disk, and later reused by CPU-only FLORA runs.
"""

import os
import math
import time
import logging
import argparse
from collections import defaultdict
import pickle
import numpy as np
import faiss
import Announce
import literal_base


#################################################################
#                FAISS embedding index helpers                  #
#################################################################

def normalize_embedding_matrix(embedding_matrix):
    """ Normalize the embedding matrix to have unit length vectors. """
    matrix = np.asarray(embedding_matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0, norms, 1.0)
    return matrix / safe_norms

def create_faiss_inner_product_index(dim, index_type='flat', hnsw_m=32, hnsw_ef_search=64, hnsw_ef_construction=200):
    """Create a FAISS IP index.
    ``flat`` uses exact-search and uses FAISS GPU when available. 
    ``hnsw`` uses approximate CPU search, which is usually much faster for large literal candidate sets.
    """
    if index_type not in {'flat', 'hnsw'}:
        raise ValueError("Unsupported FAISS literal index type: %s" % index_type)

    if index_type == 'hnsw':
        try:
            index = faiss.IndexHNSWFlat(dim, int(hnsw_m), faiss.METRIC_INNER_PRODUCT)
        except TypeError:
            index = faiss.IndexHNSWFlat(dim, int(hnsw_m))
            index.metric_type = faiss.METRIC_INNER_PRODUCT
        index.hnsw.efConstruction = int(hnsw_ef_construction)
        index.hnsw.efSearch = int(hnsw_ef_search)
        logging.info(
            "FAISS literal search using HNSW CPU index | M=%s | efSearch=%s | efConstruction=%s",
            hnsw_m, hnsw_ef_search, hnsw_ef_construction,
        )
        print(
            "   FAISS literal search using HNSW CPU index "
            "(M=%s, efSearch=%s, efConstruction=%s)." % (
                hnsw_m, hnsw_ef_search, hnsw_ef_construction,
            ),
            flush=True,
        )
        return index, None

    index = faiss.IndexFlatIP(dim)

    if not hasattr(faiss, "StandardGpuResources") or not hasattr(faiss, "index_cpu_to_gpu"):
        message = "This FAISS build use CPU."
        logging.warning(message)
        print(message, flush=True)
        return index, None

    try:
        gpu_count = faiss.get_num_gpus()
    except Exception:
        gpu_count = 0
    if gpu_count <= 0:
        logging.info("FAISS literal search using CPU.")
        print("   FAISS literal search using CPU.", flush=True)
        return index, None

    gpu_res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
    logging.info("FAISS literal search using GPU cuda:0.")
    print("   FAISS literal search using GPU cuda:0.", flush=True)
    return index, gpu_res


#################################################################
#                     IDF gating helpers                        #
#################################################################

def _iter_literal_facts(kb):
    if (hasattr(kb, 'num_entities')
        and hasattr(kb, 'triplesWithSubjectIds')
        and hasattr(kb, 'predicate_for_id')
        and hasattr(kb, 'entity_for_id')
        and hasattr(kb, 'is_literal_id')):
        for subject_id in range(kb.num_entities()):
            for _, predicate_id, object_id in kb.triplesWithSubjectIds(subject_id):
                if kb.is_literal_id(object_id):
                    yield subject_id, kb.predicate_for_id(predicate_id), kb.entity_for_id(object_id)
        return

    for subject, predicate, obj in kb:
        if literal_base.isLiteral(obj):
            yield subject, predicate, obj


def compute_literal_idf_weights(kb):
    """
    Return {literal_object: weight} using subject-level IDF.
    The weight is computed as log((N + 1) / df) / log(N + 1), 
    where N is the total number of subjects and df is the number of subjects that have the literal object.
    """
    literal_subjects = defaultdict(set)
    all_subjects = set()

    for subject, _, obj in _iter_literal_facts(kb):
        all_subjects.add(subject)
        literal_subjects[obj].add(subject)

    subject_count = len(all_subjects)
    max_idf = math.log(subject_count + 1)
    if subject_count <= 0 or max_idf <= 0:
        return {literal_obj: 1.0 for literal_obj in literal_subjects}

    weights = {}
    for literal_obj, subjects in literal_subjects.items():
        df = max(1, len(subjects))
        normalized_idf = math.log((subject_count + 1) / df) / max_idf
        weights[literal_obj] = max(0.0, min(1.0, normalized_idf))
    return weights


def update_literal_score(scores, literal1, literal2, score, weights1=None, weights2=None, min_score=None):
    """Filter a literal candidate by IDF without changing its semantic score.
    """
    gate_score = score
    if weights1 is not None and weights2 is not None:
        gate_score *= math.sqrt(weights1.get(literal1, 1.0) * weights2.get(literal2, 1.0))
    if min_score is not None and gate_score < min_score:
        return False
    row = scores.setdefault(literal1, {})
    if score > row.get(literal2, 0):
        row[literal2] = score
        return True
    return False


#################################################################
#                Bucket-level literal comparison                #
#################################################################

# Get buckets based on types
def getLiteralBuckets(kb, literal_english_filter=False):
    quantityBucket = defaultdict(list) # e.g., integers, floats
    digitBucket = defaultdict(list) # e.g., IDs, code versions, identifier
    strBucket = defaultdict(list) # e.g., strings
    dateBucket = defaultdict(list) # e.g., dates
    literal_objects = literal_base.iter_literal_objects(kb)

    for object1 in literal_objects:
        literal, numValue, lang, datatype = literal_base.splitLiteral(object1)
        if literal_base.is_punctuation_only_literal(literal):
            continue
        # dates handling
        if datatype in literal_base.DATE_DATATYPES:
            dateBucket[literal].append(object1)
            continue
        # numeric handling
        if numValue is not None:
            if isinstance(numValue, (int, float)):
                if datatype is not None: # typed quantities, including unit-bearing datatypes
                    quantity_key = literal_base.numeric_quantity_key(numValue, datatype)
                    quantityBucket[quantity_key].append(object1)
                    continue # TBD: may delete
                    # numberBucket[numValue].append(object1)
                # e.g., "0" vs "0"^^xsd:decimal is different
                # serve it as a string, e.g., code version 1.0, 01
                digitBucket[literal].append(object1)
                continue
            # xsd:normalizedString for phone numbers, post codes, etc.
            if len(numValue) > 0:
                strBucket[numValue].append(object1)
            continue
        # string handling
        if literal_english_filter and not literal_base.is_english_string_literal(literal, lang):
            continue
        # if is_human_readable(literal):
        strBucket[literal].append(object1)
        normalized_literal = literal_base.normalize_string_literal(literal)
        if normalized_literal != literal:
            strBucket[normalized_literal].append(object1)
            
    return quantityBucket, digitBucket, strBucket, dateBucket

def compareLiterals(sameAsScores, bucket1, bucket2, datatype=None, weights1=None, weights2=None, min_score=None):
    """
    Compare two literal buckets and add sameAs scores for compatible literal values.

    The buckets are produced by getLiteralBuckets(), so each key represents a
    normalized literal value and each value is the list of original literal terms
    that produced it. Numeric keys contain a dimension and canonical value;
    numeric buckets use approximate equality only within the same dimension, date
    buckets allow prefix matches such as year-month against year-month-day, and
    digit/string-like buckets require exact normalized-key equality.

    Parameters
    ----------
    sameAsScores : dict
        Nested dictionary updated in place with literal-to-literal scores.
    bucket1 : dict
        Source-side normalized literal bucket.
    bucket2 : dict
        Target-side normalized literal bucket.
    datatype : str
        Bucket type to compare. Supported values are 'quantity', 'date', and
        digit/string-like fallback values.
    weights1 : dict, optional
        Source-side literal IDF weights.
    weights2 : dict, optional
        Target-side literal IDF weights.
    min_score : float, optional
        Drop weighted scores below this threshold.
    """
    if datatype is None:
        raise ValueError('Datatype must be specified')
    # strings, dates
    if datatype == 'string':
        pass
    elif datatype == 'quantity':
        keys2_by_dimension = defaultdict(list)
        for dimension2, value2 in bucket2:
            keys2_by_dimension[dimension2].append((value2, (dimension2, value2)))
        for dimension1, value1 in bucket1:
            for value2, key2 in keys2_by_dimension.get(dimension1, ()):
                if math.isclose(value1, value2):
                    for object1 in bucket1[(dimension1, value1)]:
                        for object2 in bucket2[key2]:
                            update_literal_score(sameAsScores, object1, object2, 1.0, weights1, weights2, min_score)
    elif datatype == 'date':
        for key1 in bucket1:
            if key1 in bucket2:
                for object1 in bucket1[key1]:
                    for object2 in bucket2[key1]:
                        update_literal_score(sameAsScores, object1, object2, 1.0, weights1, weights2, min_score)
                continue
            last_dash = key1.rfind('-')
            while last_dash != -1:
                key1_ = key1[:last_dash]
                if key1_ in bucket2:
                    for object1 in bucket1[key1]:
                        for object2 in bucket2[key1_]:
                            update_literal_score(sameAsScores, object1, object2, (key1_.count('-')+1) / 3, weights1, weights2, min_score)
                    break
                last_dash = key1.rfind('-', 0, last_dash)       
    else: # digits, e.g., IDs, code versions, identifier
        for key in bucket1:
            if key in bucket2 and len(key.strip('"')) > 0:
                for object1 in bucket1[key]:
                    for object2 in bucket2[key]:
                        update_literal_score(sameAsScores, object1, object2, 1.0, weights1, weights2, min_score)

def compareLiterals_identity(sameAsScores, bucket1, bucket2, weights1=None, weights2=None, min_score=None):
    for key in bucket1:
        if key in bucket2:
            for object1 in bucket1[key]:
                for object2 in bucket2[key]:
                    update_literal_score(sameAsScores, object1, object2, 1.0, weights1, weights2, min_score)


def has_perfect_literal_match(literal, *score_maps):
    for score_map in score_maps:
        if not score_map:
            continue
        scores = score_map.get(literal)
        if scores and max(scores.values(), default=0) >= 1.0:
            return True
    return False


def all_literals_have_perfect_match(literals, *score_maps):
    return all(has_perfect_literal_match(literal, *score_maps) for literal in literals)


def _bucket_stats(bucket):
    return len(bucket), sum(len(objects) for objects in bucket.values())


#################################################################
#               Main literal matching function                  #
#################################################################

def load_emb(path):
    with open(path, 'rb') as file:
        emb = pickle.load(file)
    literal2id = emb['id']
    if 'emb_path' in emb:
        embedding_matrix = np.load(os.path.join(os.path.dirname(path), emb['emb_path']), mmap_mode='r')
    else:
        embedding_matrix = emb['emb']
    if (not isinstance(literal2id, dict) or embedding_matrix.ndim != 2
        or len(literal2id) != embedding_matrix.shape[0]
        or any(not isinstance(index, (int, np.integer)) or index < 0 or index >= len(literal2id)
               for index in literal2id.values())):
        raise ValueError("Invalid literal embedding shape or IDs: %s" % path)
    return literal2id, embedding_matrix


LITERAL_SEARCH_TOP_K = 1
LITERAL_HNSW_M = 32
LITERAL_HNSW_EF_SEARCH = 64
LITERAL_HNSW_EF_CONSTRUCTION = 200

def mapLiterals(
    kb1,
    kb2,
    path_emb,
    sameAsScore,
    literal_identity_only=False,
    threshold=0.5,
    chunk_size=32768,
    literal_english_filter=False,
    literal_idf=False,
    literal_faiss_index='flat',
):
    """
    Compute literal-based initialization scores between two knowledge bases.

    This is the main literal matching pipeline. It groups literal objects by
    datatype/normalized value, adds exact numeric/date/string matches, optionally
    applies subject-level IDF gating, and then uses FAISS over precomputed
    literal embeddings to find approximate string matches. The resulting literal
    matches are merged into sameAsScore in place.

    Parameters
    ----------
    kb1 : Graph-like
        Source knowledge base, CompactGraph, or LiteralOnlyGraph adapter.
    kb2 : Graph-like
        Target knowledge base, CompactGraph, or LiteralOnlyGraph adapter.
    path_emb : str
        Directory containing kb1.pkl and kb2.pkl literal embedding files.
    sameAsScore : dict
        Nested dictionary updated in place with literal sameAs scores.
    literal_identity_only : bool, optional
        If true, use exact literal identity only and skip embedding search.
    threshold : float, optional
        Minimum embedding similarity score for unweighted FAISS matches.
    chunk_size : int, optional
        Number of source literal embeddings queried per FAISS batch.
    literal_english_filter : bool, optional
        Keep only likely English string literals before string matching.
    literal_idf : bool, optional
        Filter candidates using subject-level literal IDF while retaining the
        original similarity of candidates that pass the gate.
    literal_faiss_index : str, optional
        FAISS index type, either 'flat' for exact search or 'hnsw' for approximate CPU search.
    """
    total_start = time.time()
    mapScores = {}
    logging.info(
        "Literal matching started | identity_only=%s | threshold=%s | chunk_size=%s | "
        "english_filter=%s | faiss_index=%s | hnsw_m=%s | hnsw_ef_search=%s | "
        "hnsw_ef_construction=%s | embedding=%s",
        literal_identity_only, threshold, chunk_size,
        literal_english_filter, literal_faiss_index, LITERAL_HNSW_M, LITERAL_HNSW_EF_SEARCH,
        LITERAL_HNSW_EF_CONSTRUCTION, path_emb,
    )
    # Build literal buckets
    stage_start = time.time()
    quantityBucket1, digitBucket1, strBucket1, dateBucket1 = getLiteralBuckets(kb1, literal_english_filter=literal_english_filter)
    quantityBucket2, digitBucket2, strBucket2, dateBucket2 = getLiteralBuckets(kb2, literal_english_filter=literal_english_filter)
    logging.info(
        "Literal buckets built | elapsed_min=%.3f | "
        "kb1 quantity=%s/%s digit=%s/%s string=%s/%s date=%s/%s | "
        "kb2 quantity=%s/%s digit=%s/%s string=%s/%s date=%s/%s",
        (time.time() - stage_start) / 60,
        *_bucket_stats(quantityBucket1), *_bucket_stats(digitBucket1), *_bucket_stats(strBucket1), *_bucket_stats(dateBucket1),
        *_bucket_stats(quantityBucket2), *_bucket_stats(digitBucket2), *_bucket_stats(strBucket2), *_bucket_stats(dateBucket2),
    )
    # Compute literal IDF weights if requested
    weights1 = weights2 = None
    if literal_idf:
        stage_start = time.time()
        weights1 = compute_literal_idf_weights(kb1)
        weights2 = compute_literal_idf_weights(kb2)
        logging.info(
            "Literal IDF weights computed | elapsed_min=%.3f | kb1=%s | kb2=%s",
            (time.time() - stage_start) / 60, len(weights1), len(weights2),
        )
    idf_gate_min_score = threshold if literal_idf else None
    if not literal_identity_only:
        # Exact matches are handled first, 
        # so that FAISS search can skip source literals that already have a perfect match.

        # Dates
        stage_start = time.time()
        compareLiterals(mapScores, dateBucket1, dateBucket2, 'date', weights1, weights2, idf_gate_min_score)
        # Compare numbers
        compareLiterals(mapScores, quantityBucket1, quantityBucket2, 'quantity', weights1, weights2, idf_gate_min_score)
        compareLiterals(mapScores, digitBucket1, digitBucket2, 'digit', weights1, weights2, idf_gate_min_score)
        logging.info(
            "Literal date/number matching done | elapsed_min=%.3f | sources=%s | pairs=%s",
            (time.time() - stage_start) / 60, len(mapScores), sum(len(values) for values in mapScores.values()),
        )
        # Compare strings
        stage_start = time.time()
        compareLiterals_identity(mapScores, strBucket1, strBucket2, weights1, weights2, idf_gate_min_score) # first get exact match
        logging.info(
            "Literal exact string matching done | elapsed_min=%.3f | sources=%s | pairs=%s",
            (time.time() - stage_start) / 60, len(mapScores), sum(len(values) for values in mapScores.values()),
        )

        # Embedding-based string matching.

        emb1_path = os.path.join(path_emb, 'kb1.pkl')
        emb2_path = os.path.join(path_emb, 'kb2.pkl')

        if os.path.exists(emb1_path) and os.path.exists(emb2_path):
            stage_start = time.time()
            literal2id_kb1, embedding_matrix_kb1 = load_emb(emb1_path)
            literal2id_kb2, embedding_matrix_kb2 = load_emb(emb2_path)
            if embedding_matrix_kb1.shape[1] != embedding_matrix_kb2.shape[1]:
                raise ValueError("KG1 and KG2 literal embeddings have different dimensions")
            logging.info(
                "Literal embeddings loaded | elapsed_min=%.3f | kb1_literals=%s | kb2_literals=%s",
                (time.time() - stage_start) / 60, len(literal2id_kb1), len(literal2id_kb2),
            )

            candidate_keys1 = []
            candidate_ids1 = []

            stage_start = time.time()
            for key1 in strBucket1:
                # Empty string
                if len(key1.strip('"')) == 0:
                    continue

                # Already handled by exact string matching.
                if key1 in strBucket2:
                    continue

                # Skip embedding search when this bucket key only contains source literals that already have a perfect match.
                if all_literals_have_perfect_match(strBucket1[key1], mapScores, sameAsScore):
                    continue

                # Keep only human-readable strings.
                if not literal_base.is_faiss_literal_candidate(key1):
                    continue

                # Must have a precomputed embedding.
                if key1 not in literal2id_kb1:
                    continue

                candidate_keys1.append(key1)
                candidate_ids1.append(literal2id_kb1[key1])

            candidate_keys2 = []
            candidate_ids2 = []

            for key2 in strBucket2:
                if len(key2.strip('"')) == 0:
                    continue

                # Keep only human-readable strings.
                if not literal_base.is_faiss_literal_candidate(key2):
                    continue

                if key2 not in literal2id_kb2:
                    continue

                candidate_keys2.append(key2)
                candidate_ids2.append(literal2id_kb2[key2])
            logging.info(
                "Literal FAISS candidates selected | elapsed_min=%.3f | kb1=%s | kb2=%s",
                (time.time() - stage_start) / 60, len(candidate_ids1), len(candidate_ids2),
            )

            if len(candidate_ids1) > 0 and len(candidate_ids2) > 0:
                candidate_ids1 = np.asarray(candidate_ids1, dtype=np.int64)
                candidate_ids2 = np.asarray(candidate_ids2, dtype=np.int64)

                # Normalize the embedding matrices.
                stage_start = time.time()
                emb1_valid = np.ascontiguousarray(normalize_embedding_matrix(embedding_matrix_kb1[candidate_ids1]), dtype=np.float32)
                emb2_valid = np.ascontiguousarray(normalize_embedding_matrix(embedding_matrix_kb2[candidate_ids2]), dtype=np.float32)
                logging.info(
                    "Literal embedding matrices prepared | elapsed_min=%.3f | emb1_shape=%s | emb2_shape=%s",
                    (time.time() - stage_start) / 60, emb1_valid.shape, emb2_valid.shape,
                )

                # Since embeddings are L2-normalized, FAISS inner product is cosine similarity.
                dim = emb2_valid.shape[1]
                # Keep gpu_res alive for the lifetime of the GPU FAISS index.
                index, gpu_res = create_faiss_inner_product_index(
                    dim,
                    index_type=literal_faiss_index,
                    hnsw_m=LITERAL_HNSW_M,
                    hnsw_ef_search=LITERAL_HNSW_EF_SEARCH,
                    hnsw_ef_construction=LITERAL_HNSW_EF_CONSTRUCTION,
                )
                stage_start = time.time()
                index.add(emb2_valid)
                logging.info(
                    "Literal FAISS index populated | elapsed_min=%.3f | vectors=%s | dim=%s",
                    (time.time() - stage_start) / 60, len(candidate_keys2), dim,
                )

                # Query in batches to avoid allocating large arrays.
                query_batch_size = chunk_size

                stage_start = time.time()
                for start1 in range(0, len(candidate_keys1), query_batch_size):
                    end1 = min(start1 + query_batch_size, len(candidate_keys1))

                    block_emb1 = np.ascontiguousarray(emb1_valid[start1:end1], dtype=np.float32)

                    local_best_scores, local_best_indices = index.search(block_emb1, LITERAL_SEARCH_TOP_K)

                    for row_idx, key1 in enumerate(candidate_keys1[start1:end1]):
                        for best_score, idx2 in zip(local_best_scores[row_idx], local_best_indices[row_idx]):
                            if idx2 < 0:
                                continue

                            best_score = float(best_score)
                            if best_score > 1.0:
                                best_score = 1.0
                            if not literal_idf and best_score < threshold:
                                continue

                            literal2 = candidate_keys2[int(idx2)]

                            for object1 in strBucket1[key1]:
                                if has_perfect_literal_match(object1, mapScores, sameAsScore):
                                    continue

                                if object1 not in mapScores:
                                    mapScores[object1] = {}

                                for object2 in strBucket2[literal2]:
                                    update_literal_score(mapScores, object1, object2, best_score, weights1, weights2, idf_gate_min_score)
                logging.info(
                    "Literal FAISS top-1 search done | elapsed_min=%.3f | sources=%s | pairs=%s",
                    (time.time() - stage_start) / 60, len(mapScores), sum(len(values) for values in mapScores.values()),
                )
        else: # no embeddings files available
            print("   Warning: No valid embedding files for literals, Using exact match only.")
    else:
        # Identity mapping only
        stage_start = time.time()
        compareLiterals_identity(mapScores, strBucket1, strBucket2, weights1, weights2, idf_gate_min_score)
        compareLiterals_identity(mapScores, dateBucket1, dateBucket2, weights1, weights2, idf_gate_min_score)
        compareLiterals_identity(mapScores, quantityBucket1, quantityBucket2, weights1, weights2, idf_gate_min_score)
        compareLiterals_identity(mapScores, digitBucket1, digitBucket2, weights1, weights2, idf_gate_min_score)
        logging.info(
            "Literal identity matching done | elapsed_min=%.3f | sources=%s | pairs=%s",
            (time.time() - stage_start) / 60, len(mapScores), sum(len(values) for values in mapScores.values()),
        )
    
    # Load to sameAsScore
    stage_start = time.time()
    for literal1 in mapScores:
        if hasattr(kb1, 'entity_id'):
            if kb1.entity_id(literal1) is None:
                continue
        elif not kb1.has_subject(literal1):
            continue
        # check empty mapping
        if literal1 not in sameAsScore:
            sameAsScore[literal1] = {}
        for literal2, score in mapScores[literal1].items():
            if score > sameAsScore[literal1].get(literal2, 0):
                sameAsScore[literal1][literal2] = score
    logging.info(
        "Literal matching loaded into sameAsScore | elapsed_min=%.3f | total_elapsed_min=%.3f | sources=%s | pairs=%s",
        (time.time() - stage_start) / 60, (time.time() - total_start) / 60, len(sameAsScore), sum(len(values) for values in sameAsScore.values()),
    )


#################################################################
#  Literal-only TTL loaders for memory-efficient matching       #
#################################################################

class LiteralOnlyGraph:
    """A Graph-like adapter backed by a reduced in-memory literal index.

    It keeps only literal objects, and optionally literal -> subjects mappings for IDF calculation, 
    so mapLiterals can run without loading the full graph into memory.
    """

    def __init__(self, literal_objects, literal_subjects=None):
        self.literal_objects = literal_objects # literal objects
        self.literal_subjects = literal_subjects # literal -> subjects mappings

    def objects(self, subject=None, predicate=None):
        # mapLiterals only needs the full literal-object set.
        if subject is None and predicate is None:
            return self.literal_objects
        return set()

    def has_subject(self, subject):
        # used to check if a literal exists
        return subject in self.literal_objects

    def iter_subjects(self, predicate=None, object=None, limit=None):
        # Subject iteration is available only when collect_subjects=True.
        if object is None or self.literal_subjects is None:
            return
        subjects = self.literal_subjects.get(object, ())
        for index, subject in enumerate(subjects):
            if limit is not None and index >= limit:
                break
            yield subject

    def __iter__(self):
        # compute_literal_idf_weights expects (subject, predicate, object).
        # Predicate values are not needed for IDF, so yields None.
        if self.literal_subjects is None:
            return
        for obj, subjects in self.literal_subjects.items():
            for subject in subjects:
                yield subject, None, obj


def load_literal_only_graph_from_ttl(path, collect_subjects=False, fast_line_parser=False):
    """Scan a TTL file and return a literal-only in-memory graph adapter.

    This avoids materializing the complete RDF graph. It stores unique literal
    objects, plus literal -> subjects mappings when IDF needs subject counts.
    """
    literal_objects = set()
    literal_subjects = defaultdict(set) if collect_subjects else None
    literal_fact_count = 0
    start_time = time.time()

    for subject, obj in literal_base.iter_ttl_literal_facts(path, fast_line_parser=fast_line_parser):
        literal_fact_count += 1
        literal_objects.add(obj)
        if literal_subjects is not None:
            literal_subjects[obj].add(subject)

    logging.info(
        "Literal-only TTL scan done | file=%s | parsed_literal_facts=%s | unique_literals=%s | elapsed_min=%.3f",
        path, literal_fact_count, len(literal_objects), (time.time() - start_time) / 60,
    )
    return LiteralOnlyGraph(literal_objects, literal_subjects)


#################################################################
#     Command-line interface for standalone precomputation      #
#################################################################

def get_params():
    parser = argparse.ArgumentParser(description="Compute FLORA literal matching scores.")
    parser.add_argument("--kg1", type=str, metavar='PATH',required=True, help="KG1 Turtle file, e.g., ../data/my_dataset/kg1.ttl")
    parser.add_argument("--kg2", type=str, metavar='PATH', required=True, help="KG2 Turtle file, e.g., ../data/my_dataset/kg2.ttl")
    parser.add_argument("--embedding", type=str, metavar='DIR', help="Folder containing kb1.pkl/kb2.pkl literal embeddings; required unless --string_identity is set")
    parser.add_argument("--output", type=str, metavar='PATH', required=True, help="Output pickle path for literal sameAs scores, e.g., ../data/literal_matching/my_dataset/literal_scores.pkl")
    parser.add_argument("--init", type=float, metavar='FLOAT', default=0.7, help="Initial literal similarity threshold; requires finite FLOAT in [0, 1]")
    parser.add_argument("--string_identity", action="store_true", help="Boolean flag : Use exact literal identity only")
    parser.add_argument("--chunk_size", type=int, metavar='INT', default=32768, help="FAISS query batch size; requires positive INT")
    parser.add_argument("--literal_english_filter", action="store_true", help="Boolean flag : keep only English literals during literal initialization")
    parser.add_argument("--literal_idf", action="store_true", help="Boolean flag : use literal IDF to filter common candidates without changing the retained similarity scores")
    parser.add_argument("--literal_faiss_index", choices=["flat", "hnsw"], metavar='{flat,hnsw}', default="flat", help='FAISS index for literal embedding search : flat is exact search and can use GPU; hnsw is approximate CPU search')
    parser.add_argument("--literal_embedding_model", type=str, default="Lihuchen/pearl_small", help="Model used to precompute the literal embeddings")
    parser.add_argument("--literal_parser", choices=["fast", "turtle"], metavar='{fast,turtle}', default="turtle",
        help="TTL literal parser used for streaming extraction: turtle uses FLORA's Turtle parser (default); fast uses a faster scanner for one-triple-per-line files",
    )
    args = parser.parse_args()
    if not 0 <= args.init <= 1:
        parser.error('--init must be finite and in [0, 1]')
    if args.chunk_size <= 0:
        parser.error('--chunk_size must be positive')
    paths = vars(args)
    for key in ('kg1', 'kg2', 'embedding', 'output'):
        if paths[key] is not None:
            if not paths[key].strip():
                parser.error('--%s must not be empty' % key)
            paths[key] = os.path.abspath(os.path.expanduser(paths[key]))
    for path in (args.kg1, args.kg2):
        if not os.path.isfile(path):
            parser.error('Input file does not exist: %s' % path)
    if args.embedding is None and not args.string_identity:
        parser.error('--embedding is required unless --string_identity is set')
    if not args.string_identity:
        for name in ('kb1.pkl', 'kb2.pkl'):
            if not os.path.isfile(os.path.join(args.embedding, name)):
                parser.error('Missing embedding file: %s' % os.path.join(args.embedding, name))
    return args

def main():
    args = get_params()
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
    sameAsScore = {}
    needs_subjects = args.literal_idf

    Announce.doing("Running literal matching precomputation")
    Announce.doing("Scanning literal facts")
    fast_line_parser = args.literal_parser == "fast"
    kb1 = load_literal_only_graph_from_ttl(args.kg1, collect_subjects=needs_subjects, fast_line_parser=fast_line_parser)
    kb2 = load_literal_only_graph_from_ttl(args.kg2, collect_subjects=needs_subjects, fast_line_parser=fast_line_parser)
    Announce.done()

    Announce.doing("Computing literal alignment scores")
    mapLiterals(
        kb1,
        kb2,
        args.embedding,
        sameAsScore,
        literal_identity_only=args.string_identity,
        threshold=args.init,
        chunk_size=args.chunk_size,
        literal_english_filter=args.literal_english_filter,
        literal_idf=args.literal_idf,
        literal_faiss_index=args.literal_faiss_index,
    )
    Announce.done()

    Announce.doing("Saving literal alignment scores")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as output_file:
        pickle.dump(sameAsScore, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(
        "Saved literal matching scores | output=%s | sources=%s | pairs=%s",
        args.output, len(sameAsScore), sum(len(values) for values in sameAsScore.values()),
    )
    Announce.done()
    Announce.done()
    Announce.message("Done")

if __name__ == "__main__":
    main()