#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${PROJECT_ROOT}/src"
cd "${SRC_DIR}"

printf '==========Running FLORA==========\n'

printf '==========Running Mini Test==========\n'
python main.py --dataset small-test/mini/ --output ../save/results/mini-test.ttl --embedding ../data/emb/mini/
python main.py --dataset small-test/restaurant/ --output ../save/results/small-test-restaurant.ttl --embedding ../data/emb/restaurant/
python main.py --dataset small-test/person/ --output ../save/results/small-test-person.ttl --embedding ../data/emb/person/

printf '==========Running Entity Alignment==========\n'
python main.py --dataset OpenEA/D_W_15K_V1/ --output ../save/results/dw-v1.ttl --embedding ../data/emb/D_W_15K_V1/
python main.py --dataset OpenEA/D_W_15K_V2/ --output ../save/results/dw-v2.ttl --embedding ../data/emb/D_W_15K_V2/
python main.py --dataset DBP15k/fr_en/ --output ../save/results/dbp15k-fr-en.ttl --embedding ../data/emb/fr_en/ --literal_idf
python main.py --dataset DBP15k/zh_en/ --output ../save/results/dbp15k-zh-en.ttl --embedding ../data/emb/zh_en/ --literal_idf
python main.py --dataset DBP15k/ja_en/ --output ../save/results/dbp15k-ja-en.ttl --embedding ../data/emb/ja_en/ --literal_idf

printf '==========Running KG Alignment on OAEI datasets==========\n'
python main.py --dataset OAEI/memoryalpha-stexpanded/ --output ../save/results/memoryalpha-stexpanded.ttl --embedding ../data/emb/memoryalpha-stexpanded/ --disable_predicate_identity_init
python main.py --dataset OAEI/starwars-swtor/ --output ../save/results/starwars-swtor.ttl --embedding ../data/emb/starwars-swtor/ --disable_predicate_identity_init

printf '==========Running KG Alignment on DBP1M datasets==========\n'
python main.py --kg1 ../data/DBP1M_with_name/de/kg1.ttl --kg2 ../data/DBP1M_with_name/de/kg2.ttl --embedding ../data/emb/DBP1M/de/ --output ../save/results/dbp1m_de_with_name.ttl --target_hub_degree_threshold 1000
python main.py --kg1 ../data/DBP1M_with_name/fr/kg1.ttl --kg2 ../data/DBP1M_with_name/fr/kg2.ttl --embedding ../data/emb/DBP1M/fr/ --output ../save/results/dbp1m_fr_with_name.ttl --target_hub_degree_threshold 1000

printf '==========Running KG Alignment on DBpedia_YAGO datasets==========\n'
python main.py --kg1 ../data/DBpedia_YAGO/dbpedia_en.ttl --kg2 ../data/DBpedia_YAGO/yago_en.ttl --embedding ../data/emb/DBpedia_YAGO/ --output ../save/results/dbpedia_yago.ttl --literal_idf --literal_english_filter --literal_faiss_index hnsw --target_hub_degree_threshold 1000 --workers 48 --bootstrap_workers 48 --subrelation_workers 48