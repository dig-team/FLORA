printf '==========Running==========\n'
python main.py --kb1 data/D_W_15K_V1/dbp_15K_V1.ttl --kb2 data/D_W_15K_V1/wd_15K_V1.ttl --save_file dw-v1.ttl --alpha 3.0 --init 0.7 
python main.py --kb1 data/D_W_15K_V1/dbp_15K_V2.ttl --kb2 data/D_W_15K_V2/wd_15K_V2.ttl --save_file dw-v2.ttl --alpha 3.0 --init 0.7
python main.py --kb1 data/DBP15k/fr_en/dbp-fr.ttl --kb2 data/DBP15k/fr_en/dbp-en.ttl --save_file dbp-fr-en.ttl --alpha 3.0 --init 0.7
python main.py --kb1 data/DBP15k/zh_en/dbp-zh.ttl --kb2 data/DBP15k/zh_en/dbp-en.ttl --save_file dbp-zh-en.ttl --alpha 3.0 --init 0.7
python main.py --kb1 data/DBP15k/ja_en/dbp-ja.ttl --kb2 data/DBP15k/ja_en/dbp-en.ttl --save_file dbp-ja-en.ttl --alpha 3.0 --init 0.7
python main.py --kb1 data/oaei/memoryalpha-stexpanded/source.ttl --kb2 data/oaei/memoryalpha-stexpanded/target.ttl --alpha 2.0 --init 0.7
python main.py --kb1 data/oaei/starwars-swtor/source.ttl --kb2 data/oaei/starwars-swtor/target.ttl --alpha 2.0 --init 0.7