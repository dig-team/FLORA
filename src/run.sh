printf '==========Running FLORA==========\n'

printf '==========Running Mini Test==========\n'
python main.py --dataset small-test/mini/ --save_file mini-test.ttl --alpha 3.0 --init 0.7
python main.py --dataset small-test/restaurant/ --save_file small-test-restaurant.ttl --alpha 3.0 --init 0.7
python main.py --dataset small-test/person/ --save_file small-test-person.ttl --alpha 3.0 --init 0.7

printf '==========Running Entity Alignment==========\n'
python main.py --dataset OpenEA/D_W_15K_V1/ --save_file dw-v1.ttl --alpha 3.0 --init 0.7
python main.py --dataset OpenEA/D_W_15K_V2/ --save_file dw-v2.ttl --alpha 3.0 --init 0.7
python main.py --dataset DBP15k/fr_en/ --save_file dbp-fr-en.ttl --alpha 3.0 --init 0.7
python main.py --dataset DBP15k/zh_en/ --save_file dbp-zh-en.ttl --alpha 3.0 --init 0.7
python main.py --dataset DBP15k/ja_en/ --save_file dbp-ja-en.ttl --alpha 3.0 --init 0.7

printf '==========Running KG Alignment on OAEI datasets==========\n'
python main.py --dataset OAEI/memoryalpha-stexpanded/ --save_file memoryalpha-stexpanded.ttl --alpha 3.0 --init 0.7
python main.py --dataset OAEI/starwars-swtor/ --save_file starwars-swtor.ttl --alpha 3.0 --init 0.7
