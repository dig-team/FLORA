"""
This file is part of FLORA, an unsupervised system for automatic knowledge graph (KG) alignment. 
The file is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0) by Yiwen Peng, Thomas Bonald, Fabian Suchanek and Lingyun Huang.

Description: Cache and checkpoint management utilities for reusable FLORA intermediate results and resumable runs.
"""

import hashlib
import gc
import logging
import multiprocessing as mp
import os
import pickle
import tempfile
import time

import alignment_base
import literal_matching
import log
import side_keys
import utils


def file_signature(paths):
    """Sign existing files by input position, path, size, and modification time."""
    signature = []
    for position, path in enumerate(paths):
        if not path or not os.path.exists(path):
            continue
        path = os.path.abspath(path)
        stat = os.stat(path)
        signature.append((position, path, stat.st_size, stat.st_mtime_ns))
    return tuple(signature)


def embedding_files(emb_path):
    return [os.path.join(emb_path, name) for name in
            ('kb1.pkl', 'kb2.pkl', 'kb1.npy', 'kb2.npy')]


def embedding_signature(emb_path):
    return file_signature(embedding_files(emb_path))


def dataset_cache_info(params, dataset_path):
    """Determine cache key and paths for a given dataset and loading options."""
    use_compact_kg = params.get('compact_kg', False)
    if params['dataset'] is not None:
        dataset_name = params['dataset'].rstrip('/')
        if 'OpenEA' in params['dataset']:
            source_files = [
                os.path.join(dataset_path, 'rel_triples_1'),
                os.path.join(dataset_path, 'rel_triples_2'),
                os.path.join(dataset_path, 'attr_triples_1'),
                os.path.join(dataset_path, 'attr_triples_2'),
                os.path.join(dataset_path, 'ent_links'),
            ]
            options = {'loader': 'openea', 'attr': True}
        elif 'DBP15k' in params['dataset']:
            source_files = [
                os.path.join(dataset_path, 'rel_ids_1'),
                os.path.join(dataset_path, 'rel_ids_2'),
                os.path.join(dataset_path, 'ent_ids_1'),
                os.path.join(dataset_path, 'ent_ids_2'),
                os.path.join(dataset_path, 'triples_1'),
                os.path.join(dataset_path, 'triples_2'),
                os.path.join(dataset_path, 'att_triples_1'),
                os.path.join(dataset_path, 'att_triples_2'),
            ]
            options = {'loader': 'dbp15k', 'attr': True, 'name': True}
        elif 'OAEI' in params['dataset']:
            source_files = [
                os.path.join(dataset_path, 'source.ttl'),
                os.path.join(dataset_path, 'target.ttl'),
            ]
            options = {'loader': 'oaei', 'format': 'ttl'}
        elif 'small-test' in params['dataset']:
            base_name = os.path.basename(os.path.normpath(dataset_path))
            source_files = [
                os.path.join(dataset_path, base_name + '1.ttl'),
                os.path.join(dataset_path, base_name + '2.ttl'),
            ]
            options = {'loader': 'small-test'}
        else:
            raise ValueError("Unknown dataset %s" % params['dataset'])
    else:
        source_files = [params['kg1'], params['kg2']]
        dataset_name = 'custom'
        options = {'loader': 'custom'}

    if use_compact_kg:
        options = dict(options)
        options['compact_kg'] = True

    signature = file_signature(source_files)
    cache_key_input = repr((dataset_name, options, signature)).encode('utf-8')
    cache_key = hashlib.sha256(cache_key_input).hexdigest()[:16]
    cache_dir = os.path.abspath(os.path.join(os.getcwd(), '../save/cache/kb'))
    cache_path = os.path.join(cache_dir, f'{cache_key}.pkl')
    compact_mmap_dir = os.path.abspath(os.path.join(os.getcwd(), '../save/cache/compact_kg', cache_key))
    return {
        'dataset_name': dataset_name,
        'options': options,
        'source_files': source_files,
        'signature': signature,
        'cache_key': cache_key,
        'cache_dir': cache_dir,
        'cache_path': cache_path,
        'compact_mmap_dir': compact_mmap_dir,
    }


def cache_path(cache_group, cache_key):
    """Return the path to a cache file for a given group and key."""
    cache_dir = os.path.abspath(os.path.join(os.getcwd(), '../save/cache', cache_group))
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f'{cache_key}.pkl')


def load_pickle_cache(path, expected_signature):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as cache_file:
            payload = pickle.load(cache_file)
        if payload.get('signature') == expected_signature:
            logging.debug("Loaded cache from %s", path)
            return payload.get('data')
        logging.debug("Cache signature changed, rebuilding cache: %s", path)
    except Exception as exc:
        logging.warning("Failed to load cache %s: %s", path, exc)
    return None


def save_pickle_cache(path, signature, data):
    try:
        with open(path, 'wb') as cache_file:
            pickle.dump(
                {'signature': signature, 'data': data},
                cache_file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logging.debug("Saved cache to %s", path)
    except Exception as exc:
        logging.warning("Failed to save cache %s: %s", path, exc)


def compact_knowledge_bases_if_requested(params, cache_info, kb1, kb2):
    """Convert parsed KGs to CompactGraph before saving/loading run caches."""
    if not params.get('compact_kg', False):
        return kb1, kb2
    os.makedirs(cache_info['compact_mmap_dir'], exist_ok=True)
    mmap_dir = tempfile.mkdtemp(prefix='build_', dir=cache_info['compact_mmap_dir'])
    logging.info("Building compact mmap KG arrays under %s", mmap_dir)
    compact_start = time.time()
    kb1 = utils.CompactGraph.from_graph(kb1, mmap_dir, 'kb1')
    gc.collect()
    kb2 = utils.CompactGraph.from_graph(kb2, mmap_dir, 'kb2')
    gc.collect()
    logging.info("Built compact mmap KG arrays in %s minutes", round((time.time() - compact_start) / 60, 5))
    return kb1, kb2


def load_or_compute_functionalities(graph, source_signature, graph_tag, gram, use_cache=True):
    """Load cached predicate functionalities, computing them in a child process on miss."""
    use_ids = hasattr(graph, 'iterFactIds') and hasattr(graph, 'num_entities')
    graph_side = side_keys.graph_predicate_side(graph, side_keys.PRED1 if graph_tag == 'kb1' else side_keys.PRED2)
    cache_signature = (
        source_signature,
        tuple(gram),
        'local_id' if use_ids else 'string_keyed',
        graph_side if use_ids else None,
    )
    if use_cache:
        cache_key = hashlib.sha256(repr((graph_tag, cache_signature)).encode('utf-8')).hexdigest()[:16]
        path = cache_path('functionalities', cache_key)
        cached_value = load_pickle_cache(path, cache_signature)
        if cached_value is not None:
            return cached_value
    else:
        logging.info("Computing functionalities for %s", graph_tag)

    # Temporary result files transfer worker output without retaining a cache.
    with tempfile.TemporaryDirectory(prefix='flora_functionalities_') as temp_dir:
        if not use_cache:
            path = os.path.join(temp_dir, 'result.pkl')
        # Compute in a worker so large temporary objects are released on exit.
        status_queue = mp.Queue(maxsize=1)
        worker = mp.Process(target=_compute_functionalities_cache_worker, args=(graph, gram, use_ids, path, cache_signature, status_queue))
        worker.start()
        _track_worker_peak(worker, "Functionality computation (%s)" % graph_tag)
        try:
            success, error = status_queue.get_nowait()
        except Exception:
            success, error = False, "no status returned"
        finally:
            status_queue.close()
            status_queue.join_thread()
        if worker.exitcode != 0 or not success:
            raise RuntimeError(
                "Functionality computation failed for %s: exitcode=%s error=%s"
                % (graph_tag, worker.exitcode, error)
            )
        cached_value = load_pickle_cache(path, cache_signature)
        if cached_value is None:
            raise RuntimeError("Functionality computation did not create a readable cache: %s" % path)
        return cached_value


def _compute_functionalities_cache_worker(graph, gram, use_ids, path, cache_signature, status_queue):
    try:
        if use_ids:
            computed_value = alignment_base.computeFunctionalitiesIds(graph, gram=gram)
        else:
            computed_value = alignment_base.computeFunctionalities(graph, gram=gram)
        save_pickle_cache(path, cache_signature, computed_value)
        status_queue.put((True, None))
    except BaseException as exc:
        logging.exception("Failed to compute functionalities in worker")
        status_queue.put((False, repr(exc)))


def compute_literal_embeddings_in_worker(kb1, kb2, emb_path, embedding_model):
    """Keep PyTorch threads and CUDA state out of the parent before FAISS forks."""
    worker = mp.Process(
        target=_compute_literal_embeddings_worker,
        args=(kb1, kb2, emb_path, embedding_model),
    )
    worker.start()
    try:
        _track_worker_peak(worker, "Literal embedding precomputation")
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join()
    if worker.exitcode != 0:
        raise RuntimeError("Literal embedding computation failed: exitcode=%s" % worker.exitcode)


def _compute_literal_embeddings_worker(kb1, kb2, emb_path, embedding_model):
    import literal_embedding
    literal_embedding.compute_literal_embeddings(kb1, kb2, emb_path, embedding_model=embedding_model)


def load_or_compute_literal_scores(kb1, kb2, emb_path, params, source_signature, use_cache=True):
    """Load cached literal matching scores, computing and side-keying them on miss."""
    use_id_keyed_scores = side_keys.can_use_id_keyed_state(kb1, kb2)
    cache_signature = (
        source_signature,
        None if params['string_identity'] else embedding_signature(emb_path),
        params['string_identity'],
        params['init'],
        params.get('literal_english_filter', False),
        params.get('literal_idf', False),
        params.get('literal_faiss_index', 'flat'),
        literal_matching.LITERAL_HNSW_M,
        literal_matching.LITERAL_HNSW_EF_SEARCH,
        literal_matching.LITERAL_HNSW_EF_CONSTRUCTION,
        'id_keyed' if use_id_keyed_scores else 'string_keyed',
    )
    if use_cache:
        cache_key = hashlib.sha256(repr(('literal_matching', cache_signature)).encode('utf-8')).hexdigest()[:16]
        path = cache_path('literal_matching', cache_key)
        cached_value = load_pickle_cache(path, cache_signature)
        if cached_value is not None:
            return cached_value
    else:
        logging.info(
            "Computing literal matching scores for %s and %s",
            side_keys.graph_side(kb1, side_keys.KB1),
            side_keys.graph_side(kb2, side_keys.KB2),
        )

    cleanup_path = None
    if not use_cache:
        fd, path = tempfile.mkstemp(prefix='flora_literal_scores_', suffix='.pkl')
        os.close(fd)
        cleanup_path = path

    # Compute in a worker so FAISS/embedding temporary allocations are released on exit.
    status_queue = mp.Queue(maxsize=1)
    worker = mp.Process(
        target=_compute_literal_scores_cache_worker,
        args=(kb1, kb2, emb_path, params, cache_signature, use_id_keyed_scores, path, status_queue),
    )
    worker.start()
    _track_worker_peak(worker, "Literal matching precomputation")
    try:
        success, error = status_queue.get_nowait()
    except Exception:
        success, error = False, "no status returned"
    finally:
        status_queue.close()
        status_queue.join_thread()
    if worker.exitcode != 0 or not success:
        if cleanup_path is not None:
            try:
                os.remove(cleanup_path)
            except OSError:
                pass
        raise RuntimeError(
            "Literal matching computation failed: exitcode=%s error=%s"
            % (worker.exitcode, error)
        )

    literal_scores = load_pickle_cache(path, cache_signature)
    if cleanup_path is not None:
        try:
            os.remove(cleanup_path)
        except OSError:
            pass
    if literal_scores is None:
        raise RuntimeError("Literal matching computation did not create a readable cache: %s" % path)
    return literal_scores


def _compute_literal_scores_cache_worker(kb1, kb2, emb_path, params, cache_signature, use_id_keyed_scores, path, status_queue):
    try:
        literal_scores = {}
        literal_matching.mapLiterals(
            kb1,
            kb2,
            emb_path,
            literal_scores,
            literal_identity_only=params['string_identity'],
            threshold=params['init'],
            literal_english_filter=params.get('literal_english_filter', False),
            literal_idf=params.get('literal_idf', False),
            literal_faiss_index=params.get('literal_faiss_index', 'flat'),
        )
        if use_id_keyed_scores:
            literal_scores = side_keys.maybe_encode_same_as_scores(literal_scores, kb1, kb2)
        save_pickle_cache(path, cache_signature, literal_scores)
        status_queue.put((True, None))
    except BaseException as exc:
        logging.exception("Failed to compute literal matching in worker")
        status_queue.put((False, repr(exc)))


def _track_worker_peak(worker, stage_label, sample_interval=log.MEMORY_SAMPLE_INTERVAL):
    if not logging.getLogger().isEnabledFor(logging.DEBUG):
        worker.join()
        return
    tracker = log.ProcessMemoryPeakTracker(
        stage_label,
        [worker.pid],
        sample_interval=sample_interval,
    ).start()
    try:
        worker.join()
        tracker.log_peak(stage_label)
    finally:
        tracker.stop()


def _load_knowledge_bases_cache(path, signature):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as cache_file:
            cached_payload = pickle.load(cache_file)
        if cached_payload.get('signature') == signature:
            logging.info("Loaded KG cache from %s", path)
            return cached_payload['kb1'], cached_payload['kb2'], cached_payload.get('gt_pairs')
        logging.info("KG cache signature changed, rebuilding cache: %s", path)
    except Exception as exc:
        logging.warning("Failed to load KG cache %s: %s", path, exc)
    return None


def _save_knowledge_bases_cache(path, cache_info, kb1, kb2, gt_pairs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, 'wb') as cache_file:
            pickle.dump(
                {
                    'signature': cache_info['signature'],
                    'kb1': kb1,
                    'kb2': kb2,
                    'gt_pairs': gt_pairs,
                },
                cache_file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logging.info("Saved KG cache to %s", path)
        return True
    except Exception as exc:
        logging.warning("Failed to save KG cache %s: %s", path, exc)
        return False


def load_knowledge_bases_with_cache(params, dataset_path, use_cache=True):
    """Load KGs with optional caching, always building compact graphs in a worker."""
    cache_info = dataset_cache_info(params, dataset_path)
    path = cache_info['cache_path']

    if use_cache:
        cached_value = _load_knowledge_bases_cache(path, cache_info['signature'])
        if cached_value is not None:
            return cached_value

    if params.get('compact_kg', False):
        with tempfile.TemporaryDirectory(prefix='flora_compact_kg_') as temp_dir:
            if not use_cache:
                path = os.path.join(temp_dir, 'result.pkl')
            # Build in a worker to release parsing and conversion allocations.
            status_queue = mp.Queue(maxsize=1)
            worker = mp.Process(
                target=_build_compact_knowledge_bases_cache_worker,
                args=(params, dataset_path, cache_info, path, status_queue),
            )
            worker.start()
            _track_worker_peak(worker, "Compact KG construction")
            try:
                success, error = status_queue.get_nowait()
            except Exception:
                success, error = False, "no status returned"
            finally:
                status_queue.close()
                status_queue.join_thread()
            if worker.exitcode != 0 or not success:
                raise RuntimeError(
                    "Compact KG build failed: exitcode=%s error=%s"
                    % (worker.exitcode, error)
                )
            cached_value = _load_knowledge_bases_cache(path, cache_info['signature'])
            if cached_value is None:
                raise RuntimeError("Compact KG build did not create a readable result: %s" % path)
            return cached_value

    kb1, kb2, gt_pairs = load_raw_knowledge_bases(params, dataset_path)

    kb1, kb2 = compact_knowledge_bases_if_requested(params, cache_info, kb1, kb2)

    if use_cache:
        _save_knowledge_bases_cache(path, cache_info, kb1, kb2, gt_pairs)
    return kb1, kb2, gt_pairs


def _build_compact_knowledge_bases_cache_worker(params, dataset_path, cache_info, path, status_queue):
    try:
        kb1, kb2, gt_pairs = load_raw_knowledge_bases(params, dataset_path)
        kb1, kb2 = compact_knowledge_bases_if_requested(params, cache_info, kb1, kb2)
        if not _save_knowledge_bases_cache(path, cache_info, kb1, kb2, gt_pairs):
            raise RuntimeError("failed to save KG cache")
        status_queue.put((True, None))
    except BaseException as exc:
        logging.exception("Failed to build compact KG cache in worker")
        status_queue.put((False, repr(exc)))


def load_raw_knowledge_bases(params, dataset_path):
    """Parse KGs from source files without reading or writing the KG pickle cache."""
    gt_pairs = None
    if params['dataset'] is not None:
        if 'OpenEA' in params['dataset']:
            kb1, kb2, gt_pairs = utils.load_openea(dataset_path, attr=True)
        elif 'DBP15k' in params['dataset']:
            kb1, kb2 = utils.load_dbp15k(dataset_path, attr=True, name=True)
        elif 'OAEI' in params['dataset']:
            kb1, kb2 = utils.load_oaei(dataset_path, format='ttl')
        elif 'small-test' in params['dataset']:
            base_name = os.path.basename(os.path.normpath(dataset_path))
            kb1 = utils.graphFromTurtleFile(os.path.join(dataset_path, base_name + '1.ttl'))
            kb2 = utils.graphFromTurtleFile(os.path.join(dataset_path, base_name + '2.ttl'))
        else:
            raise ValueError("Unknown dataset %s" % params['dataset'])
    else:
        kb1 = utils.graphFromTurtleFile(params['kg1'])
        kb2 = utils.graphFromTurtleFile(params['kg2'])
    return kb1, kb2, gt_pairs


def load_knowledge_bases(params, dataset_path, use_cache=True):
    """Load KGs through the configured cache policy, applying compact conversion."""
    if use_cache or params.get('compact_kg', False):
        return load_knowledge_bases_with_cache(params, dataset_path, use_cache=use_cache)
    logging.info("KG cache disabled; parsing raw source files")
    return load_raw_knowledge_bases(params, dataset_path)


def checkpoint_signature(params, kg_cache_info, emb_path=None, training_data_file=None):
    """Return the run identity used to decide whether a checkpoint is compatible."""
    relevant_params = {
        key: params.get(key)
        for key in (
            'dataset',
            'kg1',
            'kg2',
            'trainingdata',
            'embedding',
            'literal_scores',
            'alpha',
            'init',
            'string_identity',
            'gramN',
            'prune_min_score',
            'target_hub_degree_threshold',
            'epsilon',
            'disable_predicate_identity_init',
            'compact_kg',
            'literal_idf',
            'literal_faiss_index',
            'literal_embedding_model',
        )
    }
    if params.get('literal_english_filter', False):
        relevant_params['literal_english_filter'] = True
    if training_data_file is None and params.get('trainingdata'):
        training_data_file = params['trainingdata']
    if emb_path is None:
        emb_path = params.get('embedding') or '../data/emb/'
    literal_scores_path = params.get('literal_scores')
    return {
        'dataset_signature': kg_cache_info['signature'],
        'kg_cache_key': kg_cache_info['cache_key'],
        'params': relevant_params,
        'training_data_signature': file_signature([training_data_file]),
        'embedding_signature': None if params.get('string_identity') or literal_scores_path else embedding_signature(emb_path),
        'literal_scores_signature': file_signature([literal_scores_path]),
    }


def default_checkpoint_dir(output_path):
    output_stem = os.path.splitext(os.path.basename(output_path))[0]
    return os.path.abspath(os.path.join(os.getcwd(), '../save/checkpoints', output_stem))


def checkpoint_path(checkpoint_dir, iterations):
    return os.path.join(checkpoint_dir, f'checkpoint_iter_{iterations:04d}.pkl')


def latest_checkpoint_path(checkpoint_dir):
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return None
    candidates = [
        os.path.join(checkpoint_dir, filename)
        for filename in os.listdir(checkpoint_dir)
        if filename.startswith('checkpoint_iter_') and filename.endswith('.pkl')
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def save_checkpoint(checkpoint_dir, signature, iterations, sameAsScores, predicate2superPredicate, quasiEqvirel):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = checkpoint_path(checkpoint_dir, iterations)
    tmp_path = path + '.tmp'
    payload = {
        'signature': signature,
        'iterations': iterations,
        'sameAsScores': sameAsScores,
        'predicate2superPredicate': predicate2superPredicate,
        'quasiEqvirel': quasiEqvirel,
        'saved_at': time.time(),
    }
    try:
        with open(tmp_path, 'wb') as checkpoint_file:
            pickle.dump(payload, checkpoint_file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
        logging.info("Saved checkpoint iteration=%s path=%s", iterations, path)
    except Exception as exc:
        logging.warning("Failed to save checkpoint %s: %s", path, exc)
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def load_latest_checkpoint(checkpoint_dir, signature):
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        logging.info("No checkpoint found in %s", checkpoint_dir)
        return None
    paths = [os.path.join(checkpoint_dir, filename) for filename in os.listdir(checkpoint_dir)
             if filename.startswith('checkpoint_iter_') and filename.endswith('.pkl')]
    for path in sorted(paths, key=os.path.getmtime, reverse=True):
        try:
            with open(path, 'rb') as checkpoint_file:
                payload = pickle.load(checkpoint_file)
            if not isinstance(payload, dict) or payload.get('signature') != signature:
                logging.warning("Checkpoint signature does not match this run; ignoring path=%s", path)
                continue
            iterations = payload.get('iterations')
            if (type(iterations) is not int or iterations < 0
                or any(not isinstance(payload.get(key), dict) for key in
                       ('sameAsScores', 'predicate2superPredicate', 'quasiEqvirel'))):
                logging.warning("Incomplete checkpoint state; ignoring path=%s", path)
                continue
        except Exception as exc:
            logging.warning("Failed to load checkpoint %s: %s", path, exc)
            continue
        logging.info("Loaded checkpoint iteration=%s path=%s", iterations, path)
        return payload
    logging.info("No compatible checkpoint found in %s", checkpoint_dir)
    return None


def restore_checkpoint_state(params, checkpoint_dir, signature, kb1, kb2):
    """Load and normalize the latest compatible checkpoint state for this run."""
    if not params['resume_checkpoint']:
        return None

    checkpoint_payload = load_latest_checkpoint(checkpoint_dir, signature)
    if checkpoint_payload is None:
        return None

    sameAsScores = checkpoint_payload['sameAsScores']
    predicate2superPredicate = checkpoint_payload['predicate2superPredicate']
    quasiEqvirel = checkpoint_payload['quasiEqvirel']
    sameAsScores = side_keys.maybe_encode_same_as_scores(sameAsScores, kb1, kb2)
    predicate2superPredicate = side_keys.maybe_encode_predicate_mapping(
        predicate2superPredicate,
        kb1,
        kb2,
        preserve_mapping=alignment_base.is_dense_default_predicate_mapping,
    )
    quasiEqvirel = side_keys.maybe_encode_predicate_mapping(
        quasiEqvirel,
        kb1,
        kb2,
        preserve_mapping=alignment_base.is_dense_default_predicate_mapping,
    )
    iterations = checkpoint_payload['iterations']
    logging.info(
        "Resuming main loop from checkpoint | iteration=%s | checkpoint_dir=%s",
        iterations, checkpoint_dir,
    )
    log.log_nested_mapping_stats("Checkpoint sameAsScores", sameAsScores)
    log.log_predicate_mapping_stats("Checkpoint predicate2superPredicate stats", predicate2superPredicate)
    log.log_predicate_mapping_stats("Checkpoint quasiEqvirel stats", quasiEqvirel)
    return {
        'sameAsScores': sameAsScores,
        'predicate2superPredicate': predicate2superPredicate,
        'quasiEqvirel': quasiEqvirel,
        'iterations': iterations,
    }


def save_checkpoint_if_enabled(
        params,
        checkpoint_dir,
        signature,
        iterations,
        sameAsScores,
        predicate2superPredicate,
        quasiEqvirel,
        force=False):
    """Apply checkpoint policy and save the current run state when enabled."""
    if not params['enable_checkpoint']:
        return
    checkpoint_interval = params['checkpoint_interval']
    if force:
        if checkpoint_interval == 0:
            return
    else:
        if checkpoint_interval <= 0:
            return
        if iterations % checkpoint_interval != 0:
            return
    save_checkpoint(
        checkpoint_dir,
        signature,
        iterations,
        sameAsScores,
        predicate2superPredicate,
        quasiEqvirel,
    )