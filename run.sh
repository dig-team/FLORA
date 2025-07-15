printf '==========Running FLORA==========\n'

printf '==========Running Mini Test==========\n'
python main.py --dataset data/small-test/mini/ --save_file mini-test.ttl --alpha 3.0 --init 0.7
python main.py --dataset data/small-test/restaurant/ --save_file small-test-restaurant.ttl --alpha 3.0 --init 0.7
python main.py --dataset data/small-test/person/ --save_file small-test-person.ttl --alpha 3.0 --init 0.7

printf '==========Running Entity Alignment==========\n'
python main.py --dataset data/OpenEA/D_W_15K_V1/ --save_file dw-v1.ttl --alpha 3.0 --init 0.7
python main.py --dataset data/OpenEA/D_W_15K_V2/ --save_file dw-v2.ttl --alpha 3.0 --init 0.7
python main.py --dataset data/DBP15k/fr_en/ --save_file dbp-fr-en.ttl --alpha 3.0 --init 0.7
python main.py --dataset data/DBP15k/zh_en/ --save_file dbp-zh-en.ttl --alpha 3.0 --init 0.7
python main.py --dataset data/DBP15k/ja_en/ --save_file dbp-ja-en.ttl --alpha 3.0 --init 0.7

printf '==========Running KG Alignment on OAEI datasets==========\n'
python main.py --dataset data/oaei/memoryalpha-stexpanded/ --save_file memoryalpha-stexpanded.ttl --alpha 2.0 --init 0.7
python main.py --dataset data/oaei/starwars-swtor/ --save_file starwars-swtor.ttl --alpha 2.0 --init 0.7
