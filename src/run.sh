printf '==========Running FLORA==========\n'

printf '==========Running Mini Test==========\n'
python main.py --dataset small-test/mini/ --output mini-test.ttl --embedding emb/mini/ --alpha 3.0 --init 0.7
python main.py --dataset small-test/restaurant/ --output small-test-restaurant.ttl --embedding emb/restaurant/ --alpha 3.0 --init 0.7
python main.py --dataset small-test/person/ --output small-test-person.ttl --embedding emb/person/ --alpha 3.0 --init 0.7

printf '==========Running Entity Alignment==========\n'
python main.py --dataset OpenEA/D_W_15K_V1/ --output dw-v1.ttl --embedding emb/D_W_15K_V1/ --alpha 3.0 --init 0.7
python main.py --dataset OpenEA/D_W_15K_V2/ --output dw-v2.ttl --embedding emb/D_W_15K_V2/ --alpha 3.0 --init 0.7
python main.py --dataset DBP15k/fr_en/ --output dbp-fr-en.ttl --embedding emb/fr_en/ --alpha 3.0 --init 0.7
python main.py --dataset DBP15k/zh_en/ --output dbp-zh-en.ttl --embedding emb/zh_en/ --alpha 3.0 --init 0.7
python main.py --dataset DBP15k/ja_en/ --output dbp-ja-en.ttl --embedding emb/ja_en/ --alpha 3.0 --init 0.7

printf '==========Running KG Alignment on OAEI datasets==========\n'
python main.py --dataset OAEI/memoryalpha-stexpanded/ --output memoryalpha-stexpanded.ttl --embedding emb/memoryalpha-stexpanded/ --alpha 3.0 --init 0.7
python main.py --dataset OAEI/starwars-swtor/ --output starwars-swtor.ttl --embedding emb/starwars-swtor/ --alpha 3.0 --init 0.7
