import utils
import Prefixes
import Announce
import time
import init
import multiprocessing as mp
import argparse
import logging
import os
import sys
import align


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
            python main.py --kg1 ../data/kg1.ttl --kg2 ../data/kg2.ttl --embedding ../data/emb/ --output results.ttl

        2) For benchmark datasets, please use --dataset parameter, for example:
            python main.py --dataset OpenEA/D_W_15K_V2/ --embedding emb/D_W_15K_V2/ --alpha 3.0 --init 0.7 --output dw-v2.ttl

        To quickly test the code, you can use the small-test dataset:
            python main.py --dataset small-test/mini/ --embedding emb/mini/ --output mini-test.ttl
        """,
        formatter_class=CustomFormatter
    )
    parser.add_argument('--alpha', type=float, default=3.0, help='Benefit of doubt factor for calculating subrelation scores')
    parser.add_argument('--init', type=float, default=0.7, help='Initial literal similarity threshold')
    parser.add_argument('--gramN', type=int, default=100, help='The maximum number of evidences to consider for each entity during alignment')
    parser.add_argument('--epsilon', type=float, default=0.01, help='Convergence threshold')
    parser.add_argument('--output', type=str, default='results.ttl', help='Output file name')
    parser.add_argument('--embedding', type=str, default=None, help='Embedding folder path for the input two KGs, e.g., emb/D_W_15K_V2/')
    parser.add_argument('--trainingdata', type=str, default=None, help='Training data file name if any')
    parser.add_argument('--string_identity', type=bool, default=False, help='Whether to use string identity for literal matching')
    # Default datasets
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name for KG alignment, e.g., OpenEA/D_W_15K_V2/')
    # Optional parameters for customized datasets
    parser.add_argument('--kg1', type=str, default='../data/source.ttl', help='customized source turtle file as KG1')
    parser.add_argument('--kg2', type=str, default='../data/target.ttl', help='customized target turtle file as KG2')

    # Show help if no args
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)

    args, _ = parser.parse_known_args()
    params_ = vars(args)
    return params_



if __name__ == '__main__':
    Announce.doing("Running FLORA...")

    params = get_params()
    Announce.set_logger(params)

    # File paths
    dataset_path = '../data/{a}'.format(a=params['dataset']) if params['dataset'] else None
    training_data_file = '../data/{a}'.format(a=params['trainingdata']) if params['trainingdata'] else None
    if dataset_path is not None:
        emb_path = '../data/{a}'.format(a=params['embedding']) if params['embedding'] else '../data/emb/' # default path
    else:
        emb_path = params['embedding'] if params['embedding'] else '../data/emb/' # default path
    output_path = '../save/{a}'.format(a=params['output'])


    #################################################################
    #                    Loading data                               #
    #################################################################

    # Load knowledge bases
    Announce.doing("Loading Knowledge Bases")
    if params['dataset'] is not None:
        if 'OpenEA' in params['dataset']:
            kb1, kb2, _ = utils.load_openea(dataset_path, attr=True)
        elif 'DBP15k' in params['dataset']:
            kb1, kb2 = utils.load_dbp15k(dataset_path, attr=True, name=True)
        elif 'OAEI' in params['dataset']:
            kb1, kb2 = utils.load_oaei(dataset_path, format='ttl')
        elif 'small-test' in params['dataset']:
            kb1 = utils.graphFromTurtleFile(os.path.join(dataset_path, dataset_path.split('/')[-2]+'1.ttl'))
            kb2 = utils.graphFromTurtleFile(os.path.join(dataset_path, dataset_path.split('/')[-2]+'2.ttl'))
        else:
            raise ValueError("Unknown dataset %s" % params['dataset'])
    else: # customized input kgs
        # check existence
        assert os.path.exists(params['kg1']), "File %s does not exist!" % params['kg1']
        assert os.path.exists(params['kg2']), "File %s does not exist!" % params['kg2']
        kb1 = utils.graphFromTurtleFile(params['kg1'])
        kb2 = utils.graphFromTurtleFile(params['kg2'])
    Announce.done()

    # Load training data (if any)
    sameAsScores={}
    # if len(sys.argv)>4:
    if training_data_file is not None:
        Announce.doing("Loading training data from %s" % training_data_file)
        with open(training_data_file, "rt", encoding="utf-8") as trainingDataFile:
            for line in trainingDataFile:
                split=line.strip().split("\t")
                if split[0] not in sameAsScores:
                    sameAsScores[split[0]]={}
                sameAsScores[split[0]][split[1]]=1.0 if len(split)<3 else float(split[2])
        Announce.done()



    #################################################################
    #            Initialization + Bootstrapping                     #
    #################################################################

    Announce.doing("Initializing Subrelations")
    predicates1 = kb1.predicates()
    predicates2 = kb2.predicates()
    predicate2superPredicate=align.initializePredicateSubsumption(predicates1, predicates2, relinit=0.1)
    Announce.done()

    Announce.doing("Computing functionalities")
    functionalities1=align.computeFunctionalities(kb1, gram=[1, 2])
    functionalities2=align.computeFunctionalities(kb2, gram=[1, 2])
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
    # Precompute the literal embeddings if necessary
    if params['embedding'] is not None and \
        os.path.exists(os.path.join(emb_path, 'kb1.pkl')) and \
            os.path.exists(os.path.join(emb_path, 'kb2.pkl')):
            Announce.doing("Loading precomputed literal embeddings from %s" % emb_path)
            Announce.done()
    else:
        Announce.doing("PreComputing literal embeddings...")
        import literals
        literals.compute_literal_embeddings(kb1, kb2, emb_path)
        Announce.done()
    init.mapLiterals(kb1, kb2, emb_path, sameAsScores, params['string_identity'], params['init'])
    Announce.done()


    # Bootstrapping the entity alignment by literals
    starttime = time.time()
    Announce.doing("Bootstrapping")
    BOD = params['alpha']
    align.bootstrap_algo(kb1, kb2, sameAsScores, predicate2superPredicate, functionalities)
    ent_maxAssign = align.bilateral_max_assign(sameAsScores)
    # Subrelations
    predicate2superPredicate = {}
    align.map_subrelations(BOD, kb1, kb2, ent_maxAssign, predicate2superPredicate)
    quasiEqvirel = align.computeQuasiEqrel(kb1, kb2, predicate2superPredicate)
    Announce.done()
    logging.info("Time used for bootstrapping: %s minutes"%(round((time.time() - starttime)/60, 5)))
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
        num_cpus = mp.cpu_count()
        num_workers = min(num_cpus-1, 90)
        ent_match_tuple_queue = mgr.Queue()
        for _ in range(num_workers):
            task = mp.Process(
                target=align._match_entities_by_rules,
                args=(
                    kb1, kb2,
                    quasiEqvirel,
                    ent_queue,
                    ent_match_tuple_queue,
                    sameAsScores,
                    functionalities,
                    params,
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
        logging.info("Aligning entities: %s minutes"%(round((time.time() - starttime1)/60, 5)))

        Announce.doing("Recomputing predicate inclusions")
        starttime1 = time.time()
        ent_maxAssign = align.bilateral_max_assign(sameAsScores)
        align.map_subrelations(BOD, kb1, kb2, ent_maxAssign, predicate2superPredicate)
        quasiEqvirel = align.computeQuasiEqrel(kb1, kb2, predicate2superPredicate)
        Announce.done()
        logging.info("Aligning predicates: %s minutes"%(round((time.time() - starttime1)/60, 5)))

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
    #                       Write out results                       #
    #################################################################
    Announce.doing("Writing out results")
    with open(output_path, "wt", encoding="utf-8") as out:
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
    logging.info("Time used for the whole procedure: %s minutes"%(round((time.time() - starttime)/60, 5)))