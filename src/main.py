"""
This file is part of FLORA, an unsupervised system for automatic knowledge graph (KG) alignment. 
The file is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0) by Yiwen Peng, Thomas Bonald, Fabian Suchanek and Lingyun Huang.

Description: Command-line entry point that configures datasets, hyperparameters, and logging, and runs the whole FLORA alignment procedure.
"""

import Prefixes
import Announce
import atexit
import time
import gc
import multiprocessing as mp
import argparse
import logging
import os
import sys
import shutil
import pickle
import alignment_base
import relation_mapping
import entity_alignment
import cache
import log
import side_keys


# hyperparameters
class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawTextHelpFormatter):
    pass

def get_params():
    parser = argparse.ArgumentParser(
        usage=argparse.SUPPRESS,
        description="""\
        
        FLORA: Unsupervised Knowledge Graph Alignment by Fuzzy Logic

        Usage: 
        There are two different ways of calling FLORA.

        1) For Custom KGs, please provide the input KGs explicitly through --kg1, --kg2, for example:
            python main.py --kg1 ../data/my_dataset/kg1.ttl --kg2 ../data/my_dataset/kg2.ttl --embedding ../data/emb/my_dataset/ --output ../save/results/my_dataset.ttl

        2) For benchmark datasets, please use --dataset parameter, for example:
            python main.py --dataset OpenEA/D_W_15K_V2/ --embedding ../data/emb/D_W_15K_V2/ --output ../save/results/dw-v2.ttl

        To quickly test the code, you can use the small-test dataset:
            python main.py --kg1 ../data/small-test/mini/mini1.ttl --kg2 ../data/small-test/mini/mini2.ttl --string_identity --output ../save/results/mini-test.ttl
        """,
        formatter_class=CustomFormatter,
    )

    # Data source and run artifacts.
    io_group = parser.add_argument_group('Input and output')
    io_group.add_argument('--dataset', type=str, metavar='DATASET', default=None,
        help=(
            'Benchmark dataset name under ../data, or explicit directory path:\n'
            '  OpenEA/D_W_15K_V1/, OpenEA/D_W_15K_V2/\n'
            '  DBP15k/fr_en/, DBP15k/ja_en/, DBP15k/zh_en/\n'
            '  OAEI/memoryalpha-stexpanded/, OAEI/starwars-swtor/\n'
            '  small-test/mini/, small-test/person/, small-test/restaurant/'
        ),
    )
    io_group.add_argument('--kg1', type=str, metavar='PATH', default=None, help='Custom source Turtle file used as KG1, e.g., ../data/my_dataset/kg1.ttl')
    io_group.add_argument('--kg2', type=str, metavar='PATH', default=None, help='Custom target Turtle file used as KG2, e.g., ../data/my_dataset/kg2.ttl')
    io_group.add_argument('--embedding', type=str, metavar='DIR', default='../data/emb/', help='Literal embedding folder for the two KGs, e.g., ../data/emb/my_dataset/')
    io_group.add_argument('--literal_scores', type=str, metavar='PATH', default=None, help='Precomputed literal sameAs score pickle from literal_matching.py, e.g., ../data/literal_matching/my_dataset/literal_scores.pkl')
    io_group.add_argument('--trainingdata', type=str, metavar='PATH', default=None, help='Optional seed alignment file, e.g., ../data/my_dataset/trainingdata.ttl')
    io_group.add_argument('--output', type=str, metavar='PATH', default='../save/results/my_dataset.ttl', help='Output file path; a bare filename is saved under ../save/results/, e.g., --output my_dataset.ttl')
    # Literal initialization
    literal_group = parser.add_argument_group('Literal initialization')
    literal_group.add_argument('--init', type=float, metavar='FLOAT', default=0.7, help='Initial literal similarity threshold; requires finite FLOAT in [0, 1]')
    literal_group.add_argument('--string_identity', action='store_true', help='Boolean flag: use only exact string literal identity for initialization; otherwise use literal embedding similarity')
    literal_group.add_argument('--literal_embedding_model', type=str, metavar='MODEL', default='Lihuchen/pearl_small', help='HuggingFace model used when FLORA needs to pre-compute literal embeddings; requires MODEL. For monolingual embeddings, use Lihuchen/pearl_small. For multilingual embeddings, use sentence-transformers/LaBSE')
    literal_group.add_argument('--literal_english_filter', action='store_true', help='Boolean flag: keep only English literals during literal initialization, for noisy English-only datasets')
    literal_group.add_argument('--literal_idf', action='store_true', help='Boolean flag: use IDF to filter high-frequency literals during literal initialization')
    literal_group.add_argument('--literal_faiss_index', choices=['flat', 'hnsw'], metavar='{flat,hnsw}', default='flat', help='FAISS index for literal embedding search: flat is exact search and can use GPU; hnsw is approximate CPU search')
    # Core alignment algorithm
    alignment_group = parser.add_argument_group('Alignment algorithm')
    alignment_group.add_argument('--alpha', type=float, metavar='FLOAT', default=3.0, help='Benefit-of-doubt factor for calculating subrelation scores; requires finite FLOAT > 0')
    alignment_group.add_argument('--gramN', type=int, metavar='INT', default=100, help='Maximum number of evidences to consider for each entity during alignment; requires positive INT')
    alignment_group.add_argument('--epsilon', type=float, metavar='FLOAT', default=0.01, help='Convergence threshold for stopping the main loop; requires finite FLOAT >= 0')
    alignment_group.add_argument('--max_iterations', type=int, metavar='INT', default=50, help='Maximum number of main-loop iterations to run; requires INT >= 0')
    alignment_group.add_argument('--prune_min_score', type=float, metavar='FLOAT', default=0.01, help='Drop sameAs candidates below this score; requires finite FLOAT in [0, 1]')
    alignment_group.add_argument('--disable_predicate_identity_init', action='store_true', help='Boolean flag: not initialize predicates with identical URIs as equivalent, start subrelation alignment from empty set')
    # Performance and memory controls
    performance_group = parser.add_argument_group('Performance and memory')
    performance_group.add_argument('--target_hub_degree_threshold', type=int, metavar='INT', default=0, help='Apply target-side hub local-functionality pruning when a candidate subject/predicate has at least this many objects; requires INT >= 0. 0 disables pruning')
    performance_group.add_argument('--bootstrap_workers', type=int, metavar='INT', default=None, help='Worker processes for bootstrap alignment; requires positive INT')
    performance_group.add_argument('--workers', type=int, metavar='INT', default=None, help='Worker processes for iteration-time entity alignment; requires positive INT')
    performance_group.add_argument('--subrelation_workers', type=int, metavar='INT', default=None, help='Worker processes for predicate subrelation mapping; requires positive INT')
    performance_group.add_argument('--compact_kg', type=str.lower, choices=['true', 'false'], default='true', help='Use compact read-only mmap graph storage; false uses ordinary Python Graph storage')
    # Cache and checkpointing
    state_group = parser.add_argument_group('Cache and checkpointing')
    state_group.add_argument('--enable_preprocessing_cache', action='store_true', help='Enable reusable preprocessing caches for KG loading, functionalities, and literal matching, which can be useful for repeated runs on the same dataset.')
    state_group.add_argument('--enable_checkpoint', action='store_true', help='Boolean flag : save resumable FLORA checkpoints; disabled by default because checkpoints can be large')
    state_group.add_argument('--checkpoint_dir', type=str, metavar='DIR', default=None, help='Directory for FLORA checkpoints, e.g., ../save/checkpoints/my_dataset/')
    state_group.add_argument('--checkpoint_interval', type=int, metavar='INT', default=1, help='When --enable_checkpoint is set, save every N completed iterations; requires INT >= 0. 0 disables periodic checkpoints')
    state_group.add_argument('--resume_checkpoint', action='store_true', help='Boolean flag : resume from the latest compatible checkpoint in --checkpoint_dir')

    # Show help if no args
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    params_ = vars(args)
    params_['compact_kg'] = args.compact_kg == 'true'
    for key in ('init', 'prune_min_score'):
        if not 0 <= params_[key] <= 1:
            parser.error('--%s must be in [0, 1]' % key)
    for key in ('alpha', 'gramN', 'workers', 'bootstrap_workers', 'subrelation_workers'):
        if params_[key] is not None and params_[key] <= 0:
            parser.error('--%s must be positive' % key)
    for key in ('epsilon', 'max_iterations', 'target_hub_degree_threshold', 'checkpoint_interval'):
        if params_[key] < 0:
            parser.error('--%s must be nonnegative' % key)

    if args.dataset and args.dataset.split('/')[0] in ('OpenEA', 'DBP15k', 'OAEI', 'small-test'):
        params_['dataset'] = os.path.join('../data', args.dataset)
    for key in ('dataset', 'kg1', 'kg2', 'embedding', 'trainingdata', 'literal_scores', 'output', 'checkpoint_dir'):
        if params_[key] is not None:
            if not params_[key].strip():
                parser.error('--%s must not be empty' % key)
            path = os.path.expanduser(params_[key])
            if key == 'output' and not os.path.dirname(path):
                path = os.path.join('../save/results', path)
            params_[key] = os.path.abspath(path)
    if params_['dataset'] is not None:
        if params_['kg1'] is not None or params_['kg2'] is not None:
            parser.error('--dataset cannot be combined with --kg1 or --kg2')
        try:
            source_files = cache.dataset_cache_info(params_, params_['dataset'])['source_files']
        except ValueError as exc:
            parser.error(str(exc))
    else:
        if params_['kg1'] is None or params_['kg2'] is None:
            parser.error('provide --dataset or both --kg1 and --kg2')
        source_files = [params_['kg1'], params_['kg2']]
    for path in source_files + [params_['trainingdata'], params_['literal_scores']]:
        if path is not None and not os.path.isfile(path):
            parser.error('Input file does not exist: %s' % path)
    return params_



if __name__ == '__main__':
    params = get_params()
    Announce.doing("Running FLORA")
    Announce.set_logger(params)
    procedure_start = time.time()

    # File paths
    dataset_path = params['dataset']
    training_data_file = params['trainingdata']
    emb_path = params['embedding']
    output_path = params['output']
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = (os.path.abspath(params['checkpoint_dir']) if params['checkpoint_dir']
        else cache.default_checkpoint_dir(output_path))


    #################################################################
    #                    Loading data                               #
    #################################################################

    # Load knowledge bases
    Announce.doing("Loading knowledge bases")
    loading_start = time.time()
    kb1, kb2, _ = cache.load_knowledge_bases(
        params,
        dataset_path,
        use_cache=params['enable_preprocessing_cache'],
    )
    side_keys.tag_graph_sides(kb1, kb2)
    logging.info("Time used for loading KGs: %s minutes"%(round((time.time() - loading_start)/60, 5)))
    Announce.done()
    # cache and checkpoint information
    kg_cache_info = cache.dataset_cache_info(params, dataset_path)
    checkpoint_signature = cache.checkpoint_signature(params, kg_cache_info, emb_path, training_data_file)
    # unique tag for the current run, allows multiple runs to share the same checkpoint dir without interfering with each other
    run_tag = "%s-pid%s-%s" % (
        os.path.splitext(os.path.basename(params['output']))[0],
        os.getpid(),
        time.strftime("%Y%m%d%H%M%S"),
    )
    compact_entity_assign_run_dir = os.path.join(
        kg_cache_info['compact_mmap_dir'],
        'entity_assign_runs',
        run_tag,
    )
    atexit.register(shutil.rmtree, compact_entity_assign_run_dir, ignore_errors=True)
    #logging.debug("Compact entity assignment runtime dir | dir=%s", compact_entity_assign_run_dir)

    # Load training data (if any)
    sameAsScores={}
    if training_data_file is not None:
        Announce.doing("Loading training data from %s" % training_data_file)
        with open(training_data_file, "rt", encoding="utf-8") as trainingDataFile:
            for line in trainingDataFile:
                split=line.strip().split("\t")
                if split[0] not in sameAsScores:
                    sameAsScores[split[0]]={}
                sameAsScores[split[0]][split[1]]=1.0 if len(split)<3 else float(split[2])
        sameAsScores = side_keys.maybe_encode_same_as_scores(sameAsScores, kb1, kb2)
        Announce.done()


    #################################################################
    #            Initialization + Bootstrapping                     #
    #################################################################

    Announce.doing("Initializing subrelations")
    predicates1 = kb1.predicates()
    predicates2 = kb2.predicates()
    if alignment_base._can_use_compact_id_match(kb1, kb2):
        predicate2superPredicate = alignment_base.CompactDenseDefaultPredicateMapping(kb1, kb2, relinit=0.1)
    else:
        predicate2superPredicate = alignment_base.initializePredicateSubsumption(predicates1, predicates2, relinit=0.1)
    Announce.done()

    compute_functionalities_start = time.time()
    Announce.doing("Computing functionalities")
    functionalities1 = cache.load_or_compute_functionalities(
        kb1,
        kg_cache_info['signature'],
        'kb1',
        [1, 2],
        use_cache=params['enable_preprocessing_cache'],
    )
    functionalities2 = cache.load_or_compute_functionalities(
        kb2,
        kg_cache_info['signature'],
        'kb2',
        [1, 2],
        use_cache=params['enable_preprocessing_cache'],
    )
    if side_keys.can_use_id_keyed_state(kb1, kb2):
        functionalities = side_keys.preencode_compact_worker_functionalities(functionalities1, functionalities2)
    else:
        functionalities = {}
        for pred in functionalities1:
            functionalities[pred] = functionalities1[pred]
        for pred in functionalities2:
            if pred not in functionalities:
                functionalities[pred] = functionalities2[pred]
                continue
            functionalities[pred] = min(functionalities[pred], functionalities2[pred])
    # release memory
    del functionalities1
    del functionalities2
    gc.collect()
    logging.info("Time used for computing functionalities: %s minutes"%(round((time.time() - compute_functionalities_start)/60, 5)))
    Announce.done()

    BOD = params['alpha']
    iterations = 0
    checkpoint_state = cache.restore_checkpoint_state(params, checkpoint_dir, checkpoint_signature, kb1, kb2)

    if checkpoint_state is not None: # load checkpoint
        sameAsScores = checkpoint_state['sameAsScores']
        predicate2superPredicate = checkpoint_state['predicate2superPredicate']
        quasiEqvirel = checkpoint_state['quasiEqvirel']
        iterations = checkpoint_state['iterations']
    else:
        # Literal Matching
        Announce.doing(
            "Computing initialization scores | literal_mode=%s | threshold=%s"
            % ('identity' if params['string_identity'] else 'embedding similarity', params['init'])
        )
        literal_matching_start = time.time()
        if params['literal_scores']: # have precomputed literal scores
            literal_scores_path = os.path.abspath(params['literal_scores'])
            Announce.doing("Loading precomputed literal scores from %s" % literal_scores_path)
            with open(literal_scores_path, "rb") as literal_scores_file:
                literal_scores = pickle.load(literal_scores_file)
            logging.info(
                "Loaded precomputed literal scores | path=%s | sources=%s | pairs=%s",
                literal_scores_path, len(literal_scores), sum(len(values) for values in literal_scores.values()),
            )
            Announce.done()
        else:
            # Precompute the literal embeddings if necessary
            if params['string_identity']:
                logging.info("Skipping literal embedding precomputation because literal identity-only matching is enabled")
            elif os.path.exists(os.path.join(emb_path, 'kb1.pkl')) and os.path.exists(os.path.join(emb_path, 'kb2.pkl')):
                Announce.doing("Loading precomputed literal embeddings from %s" % emb_path)
                Announce.done()
            else:
                Announce.doing("PreComputing literal embeddings with %s..." % params['literal_embedding_model'])
                cache.compute_literal_embeddings_in_worker(
                    kb1,
                    kb2,
                    emb_path,
                    embedding_model=params['literal_embedding_model'],
                )
                Announce.done()
            literal_scores = cache.load_or_compute_literal_scores(
                kb1,
                kb2,
                emb_path,
                params,
                kg_cache_info['signature'],
                use_cache=params['enable_preprocessing_cache'],
            )
        # Automatic embedding generation may have changed the input artifacts.
        checkpoint_signature = cache.checkpoint_signature(params, kg_cache_info, emb_path, training_data_file)
        literal_scores = side_keys.maybe_encode_same_as_scores(literal_scores, kb1, kb2)
        if sameAsScores:
            for entity1, entity2scores in literal_scores.items():
                if entity1 not in sameAsScores:
                    sameAsScores[entity1] = {}
                for entity2, score in entity2scores.items():
                    if score > sameAsScores[entity1].get(entity2, 0):
                        sameAsScores[entity1][entity2] = score
        else:
            sameAsScores = literal_scores
        del literal_scores # release memory
        gc.collect()
        logging.info("Time used for initialization matching: %s minutes"%(round((time.time() - literal_matching_start)/60, 5)))
        Announce.done()


        # Bootstrap entity alignment from the configured initialization scores.
        starttime = time.time()
        Announce.doing("Bootstrapping")
        log.log_nested_mapping_stats("Bootstrap pre-state sameAsScores", sameAsScores)
        bootstrap_workers = max(1, params['bootstrap_workers'] or mp.cpu_count())
        round_prefix = "Bootstrap"
        ent_maxAssign = None
        quasiEqvirel = None
        if sameAsScores:
            entity_alignment.bootstrap_algo(
                kb1,
                kb2,
                sameAsScores,
                predicate2superPredicate,
                functionalities,
                min_score=params['prune_min_score'],
                num_workers=bootstrap_workers,
            )
        else:
            logging.info("Skipping %s worker-stage because literal bootstrap sameAsScores is empty", round_prefix)
        log.log_nested_mapping_stats("%s post-bootstrap sameAsScores" % round_prefix, sameAsScores)
        log.log_memory_snapshot("%s post-bootstrap memory" % round_prefix)
        ent_maxAssign = alignment_base.bilateral_max_assign(sameAsScores)
        log.log_nested_mapping_stats("%s bilateral_max_assign stats" % round_prefix, ent_maxAssign)
        log.log_alignment_fanout("%s bilateral_max_assign" % round_prefix, ent_maxAssign, kb1, kb2)
        log.log_memory_snapshot("%s post-bilateral memory" % round_prefix)
        logging.info("%s bilateral max assign successfully computed.", round_prefix)

        if params['disable_predicate_identity_init']:
            predicate2superPredicate = {}
            logging.info("%s predicate identity initialization disabled; starting subrelation mapping from empty evidence.", round_prefix)
        elif alignment_base._can_use_compact_id_match(kb1, kb2):
            predicate2superPredicate = alignment_base.initializePredicateIdentityOnlyIds(kb1, kb2)
        else:
            predicate2superPredicate = alignment_base.initializePredicateIdentityOnly(predicates1, predicates2)
        relation_mapping.map_subrelations(
            BOD,kb1,kb2, ent_maxAssign, predicate2superPredicate,
            stage_label="%s subrelation-stage" % round_prefix,
            num_workers=max(1, params['subrelation_workers'] or mp.cpu_count()),
        )
        log.log_predicate_mapping_stats("%s predicate2superPredicate stats" % round_prefix, predicate2superPredicate)
        log.log_memory_snapshot("%s post-map_subrelations memory" % round_prefix)
        logging.info("%s subrelation mapping successfully computed.", round_prefix)
        quasiEqvirel = alignment_base.computeQuasiEqrel(kb1, kb2, predicate2superPredicate)
        log.log_predicate_mapping_stats("%s quasiEqvirel stats" % round_prefix, quasiEqvirel)
        log.log_memory_snapshot("%s post-quasiEqvirel memory" % round_prefix)
        logging.info("%s quasi-equivalence relation successfully computed.", round_prefix)
        Announce.done()
        logging.info("Time used for bootstrapping: %s minutes"%(round((time.time() - starttime)/60, 5)))
        cache.save_checkpoint_if_enabled(
            params,
            checkpoint_dir,
            checkpoint_signature,
            iterations,
            sameAsScores,
            predicate2superPredicate,
            quasiEqvirel,
            force=True,
        )
    logging.info("---------------Main Loop---------------")


    #################################################################
    #                         Main Loop                             #
    #################################################################


    max_iterations = max(0, params.get('max_iterations', 100))
    logging.info("Main loop max_iterations=%s | epsilon=%s", max_iterations, params['epsilon'])
    while iterations < max_iterations:
        Announce.doing("Iteration", iterations+1)
        logging.info("----Iteration %s----" % (iterations + 1))
        
        sameAsSum=sum(val for dict_ in sameAsScores.values() for val in dict_.values())

        # Entity Alignment
        Announce.doing("Applying the Entity Alignment rules")
        starttime1 = time.time()
        worker_stage_start = time.perf_counter()
        log.log_nested_mapping_stats("Iteration pre-worker sameAsScores", sameAsScores)
        log.log_alignment_fanout(
            f"Iteration {iterations + 1} pre-worker sameAsScores",
            sameAsScores,
            kb1,
            kb2,
            min_targets=20,
        )
        log.log_memory_snapshot("Iteration pre-worker memory")
        ent_maxAssign = alignment_base.bilateral_max_assign(sameAsScores)
        log.log_nested_mapping_stats("Iteration pre-worker ent_maxAssign", ent_maxAssign)
        log.log_alignment_fanout(f"Iteration {iterations + 1} pre-worker ent_maxAssign", ent_maxAssign, kb1, kb2)
        log.log_memory_snapshot("Iteration post-bilateral memory")
        use_compact_id_score_chunks = alignment_base._can_use_compact_id_match(kb1, kb2)
        # Worker parameters for the entity alignment stage.
        worker_params = params
        worker_ent_max_assign = ent_maxAssign
        merge_guard_ent_max_assign = ent_maxAssign
        merge_guard_src_max_scores = None
        merge_guard_dst_max_scores = None
        merge_guard_index = None
        assign_index_dir = None
        if use_compact_id_score_chunks:
            # Compact KG workers share the current max assignment through mmap arrays.
            worker_params = dict(params)
            assign_index_dir = os.path.join(compact_entity_assign_run_dir, f"iter_{iterations + 1:04d}")
            entity_assign_metadata = alignment_base.build_compact_entity_assign_index(
                kb1,
                kb2,
                ent_maxAssign,
                assign_index_dir,
            )
            worker_params['_compact_entity_assign_index'] = entity_assign_metadata
            # Keep only the compact max-score guards needed by the parent merge.
            merge_guard_index = alignment_base.CompactEntityAssignIndex(entity_assign_metadata)
            merge_guard_src_max_scores = merge_guard_index.src_max
            merge_guard_dst_max_scores = merge_guard_index.dst_max
            worker_ent_max_assign = None
            ent_maxAssign = None
            merge_guard_ent_max_assign = None
            gc.collect()
            # logging.debug(
            #     "Iteration %s worker score transport | mode=compact-id-chunks | "
            #     "entity_assign=shared-array",
            #     iterations + 1,
            # )
        tasks = []
        ent_queue = None
        ent_match_tuple_queue = None
        profile_queue = None
        memory_tracker = None
        try:
            log.prepare_for_worker_fork(f"Iteration {iterations + 1} worker-stage")
            num_workers = max(1, params['workers'] or mp.cpu_count())
            logging.info("Iteration %s worker config | workers=%s", iterations + 1, num_workers)
            queue_size = max(4, num_workers * 2)
            ent_queue = mp.Queue(maxsize=queue_size)
            ent_match_tuple_queue = mp.Queue(maxsize=queue_size)
            profile_queue = mp.Queue()
            merge_guard_score_cache = {}

            def merge_worker_chunk(ent_match_score_dict):
                if use_compact_id_score_chunks:
                    alignment_base.merge_same_as_score_chunk_ids(
                        sameAsScores,
                        ent_match_score_dict,
                        kb1,
                        kb2,
                        min_score=params['prune_min_score'],
                        src_max_scores=merge_guard_src_max_scores,
                        dst_max_scores=merge_guard_dst_max_scores,
                    )
                else:
                    alignment_base.merge_same_as_score_chunk(
                        sameAsScores,
                        ent_match_score_dict,
                        min_score=params['prune_min_score'],
                        ent_max_assign=merge_guard_ent_max_assign,
                        ent_max_score_cache=merge_guard_score_cache,
                    )

            for _ in range(num_workers):
                worker_args = (
                    kb1, kb2,
                    quasiEqvirel,
                    ent_queue,
                    ent_match_tuple_queue,
                    worker_ent_max_assign,
                    functionalities,
                    worker_params,
                    profile_queue,
                    )
                task = mp.Process(
                    target=log.run_worker_with_crash_logging,
                    args=(
                        entity_alignment._match_entities_by_rules,
                        worker_args,
                        profile_queue,
                        f"Iteration {iterations + 1} worker-stage",
                    ),
                )
                task.start()
                tasks.append(task)
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                memory_tracker = log.ProcessMemoryPeakTracker(
                    f"Iteration {iterations + 1} worker-stage", [task.pid for task in tasks],
                )
                memory_tracker.log_current(f"Iteration {iterations + 1} worker-stage memory start")
                memory_tracker.start()
            worker_profiles = []

            alignment_base.feed_entity_chunks(
                ent_queue,
                kb1,
                num_workers,
                result_queue=ent_match_tuple_queue,
                merge_fn=merge_worker_chunk,
                stage_label=f"Iteration {iterations + 1} worker-stage",
                side_queues=[
                    ('profile_chunks', profile_queue, worker_profiles.append),
                ],
                worker_tasks=tasks,
            )
            alignment_base.wait_for_workers_and_drain(
                tasks,
                ent_match_tuple_queue,
                merge_worker_chunk,
                stage_label=f"Iteration {iterations + 1} worker-stage",
                side_queues=[
                    ('profile_chunks', profile_queue, worker_profiles.append),
                ],
            )
            if memory_tracker is not None:
                memory_tracker.log_peak()

            logging.info("All worker processes finished. Merging results...")
            merge_stage_start = time.perf_counter()
            alignment_base.drain_queue_items(ent_match_tuple_queue, merge_worker_chunk)
            merge_stage_time = time.perf_counter() - merge_stage_start
            logging.info("Merging worker results successfully completed.")

            alignment_base.drain_queue_items(profile_queue, worker_profiles.append)
        finally:
            if memory_tracker is not None:
                memory_tracker.stop()
            if ent_queue is not None:
                ent_queue.close()
                ent_queue.join_thread()
            if ent_match_tuple_queue is not None:
                ent_match_tuple_queue.close()
                ent_match_tuple_queue.join_thread()
            if profile_queue is not None:
                profile_queue.close()
                profile_queue.join_thread()
            merge_guard_src_max_scores = None
            merge_guard_dst_max_scores = None
            merge_guard_index = None
            if assign_index_dir is not None:
                shutil.rmtree(assign_index_dir, ignore_errors=True)
        worker_stage_time = time.perf_counter() - worker_stage_start
        Announce.done()
        logging.info("Aligning entities: %s minutes"%(round((time.time() - starttime1)/60, 5)))
        log.log_nested_mapping_stats("Iteration post-worker sameAsScores", sameAsScores)
        log.log_memory_snapshot("Iteration post-worker memory")
        log.log_worker_profiles(worker_profiles)
        # Predicate Alignment
        Announce.doing("Recomputing predicate inclusions")
        starttime1 = time.time()
        bilateral_stage_start = time.perf_counter()
        ent_maxAssign = alignment_base.bilateral_max_assign(sameAsScores)
        bilateral_stage_time = time.perf_counter() - bilateral_stage_start
        log.log_nested_mapping_stats("Iteration predicate-stage ent_maxAssign", ent_maxAssign)
        log.log_alignment_fanout(f"Iteration {iterations + 1} predicate-stage ent_maxAssign", ent_maxAssign, kb1, kb2)
        log.log_memory_snapshot("Iteration predicate-stage post-bilateral memory")
        subrelation_stage_start = time.perf_counter()
        relation_mapping.map_subrelations(
            BOD,
            kb1,
            kb2,
            ent_maxAssign,
            predicate2superPredicate,
            stage_label=f"Iteration {iterations + 1} subrelation-stage",
            num_workers=max(1, params['subrelation_workers'] or mp.cpu_count()),
        )
        subrelation_stage_time = time.perf_counter() - subrelation_stage_start
        log.log_predicate_mapping_stats("Iteration predicate2superPredicate stats", predicate2superPredicate)
        log.log_memory_snapshot("Iteration predicate-stage post-map_subrelations memory")
        quasi_stage_start = time.perf_counter()
        quasiEqvirel = alignment_base.computeQuasiEqrel(kb1, kb2, predicate2superPredicate)
        quasi_stage_time = time.perf_counter() - quasi_stage_start
        log.log_predicate_mapping_stats("Iteration quasiEqvirel stats", quasiEqvirel)
        log.log_memory_snapshot("Iteration predicate-stage post-quasiEqvirel memory")
        Announce.done()
        logging.info("Aligning predicates: %s minutes"%(round((time.time() - starttime1)/60, 5)))
        logging.info(
            "Iteration timing | worker=%.3fs | merge=%.3fs | bilateral=%.3fs | subrelations=%.3fs | quasi=%.3fs",
            worker_stage_time, merge_stage_time, bilateral_stage_time, subrelation_stage_time, quasi_stage_time,
        )
        # Check convergence
        Announce.doing("Checking convergence")
        newSameAsSum=sum(val for dict_ in sameAsScores.values() for val in dict_.values())   
        Announce.done(sameAsSum,newSameAsSum)
        logging.info("SameAs sum: %s -> %s"%(sameAsSum, newSameAsSum))
        
        Announce.done() 
        iterations+=1
        cache.save_checkpoint_if_enabled(
            params,
            checkpoint_dir,
            checkpoint_signature,
            iterations,
            sameAsScores,
            predicate2superPredicate,
            quasiEqvirel,
        )
        if abs(newSameAsSum - sameAsSum) < params['epsilon']:
            logging.info(
                "Stopping after iteration %s due to convergence: delta=%s < epsilon=%s",
                iterations, abs(newSameAsSum - sameAsSum), params['epsilon'],
            )
            break
    else:
        logging.info("Stopping after reaching max_iterations=%s", max_iterations)

    #################################################################
    #                       Write out results                       #
    #################################################################
    Announce.doing("Writing out results")
    with open(output_path, "wt", encoding="utf-8") as out:
        for p, uri in (Prefixes.prefixes_dbp | Prefixes.prefixes).items():
            out.write("@prefix "+p+": <"+uri+"> .\n")
        # Predicates
        kb1_predicates=kb1.predicates()
        kb2_predicates=kb2.predicates()
        predicates = kb1_predicates | kb2_predicates
        for predicate1 in predicates:
            predicate1_keys = [predicate1]
            if side_keys.can_use_id_keyed_state(kb1, kb2):
                predicate1_keys = side_keys.predicate_side_keys(predicate1, kb1, kb2)
            for predicate1_key in predicate1_keys:
                if predicate1_key not in predicate2superPredicate:
                    continue
                for predicate2_key, score in predicate2superPredicate[predicate1_key].items():
                    if score > 0.1:
                        predicate2 = side_keys.decode_predicate_key(predicate2_key, kb1, kb2)
                        out.write(predicate1+"\trdfs:subPropertyOf\t"+predicate2+"\t.#\t"+str(score)+"\n")
        # Literals and instances
        for entity1 in sameAsScores:
            for entity2 in sameAsScores[entity1]:
                if sameAsScores[entity1][entity2] > 0: # first report all possible scores
                    entity1_out = side_keys.decode_entity_key(entity1, kb1, kb2)
                    entity2_out = side_keys.decode_entity_key(entity2, kb1, kb2)
                    out.write(entity1_out+"\towl:sameAs\t"+entity2_out+"\t.#\t"+str(sameAsScores[entity1][entity2])+"\n")
    Announce.done()
    logging.info("Time used for the whole procedure: %s minutes"%(round((time.time() - procedure_start)/60, 5)))
    shutil.rmtree(compact_entity_assign_run_dir, ignore_errors=True)
    Announce.message("Done")