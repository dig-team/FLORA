# FLORA

> This repository contains the source code and datasets for paper “FLORA: Unsupervised Knowledge Graph Alignment by Fuzzy Logic”.
> 

## Installation

To install the required packages needed, run the following command:

```bash
pip install -r requirements
```

## Dataset

FLORA uses multiple datasets from different sources:

- [OpenEA](https://github.com/nju-websoft/OpenEA): D_W_15K_V1 and D_W_15K_V2
- [DBP15K](https://github.com/nju-websoft/JAPE): fr_en, ja_en, zh_en
- [OAEI KG Track](https://oaei.ontologymatching.org/2024/knowledgegraph/index.html): memoryalpha-stexpanded, starwars-swtor

We also provide two mini-test datasets: [Person, Restaurant](https://oaei.ontologymatching.org/2010/im/index.html) from OAEI 2010 for quick test.

For detailed statistics on each dataset, refer to `statistics.pdf` . Literal embeddings are provided as xxx.pkl files in the `data/emb/` folder of each dataset. To reproduce these embeddings, use: 

```bash
python literals.py <kb1.ttl> <kb2.ttl> <emb_path>
```

## Running the Code

To produce the alignment results, your could use the following command:

```bash
python main.py -kb1 dbp-fr.ttl -kb2 dbp-en.ttl -alpha 3.0 -init 0.7
```

Or you could directly run `bash run.sh` to reproduce the results.

- Evaluation and Analysis

The final alignment results are stored in the `save` folder. To evaluate and analyze results, run `analysis.ipynb` block by block, adjusting gold standard path `REF_PATH` and results path `RES_PATH` as necessary.