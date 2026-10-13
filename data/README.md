# Datasets and Literal Embeddings

The standard benchmarks are available from their original sources: [OpenEA](https://github.com/nju-websoft/OpenEA), [DBP15K](https://github.com/nju-websoft/JAPE), [OAEI KG Track](https://oaei.ontologymatching.org/2024/knowledgegraph/index.html), and [DBP1M](https://github.com/ZJU-DBL/LargeEA).


You can also download all datasets used in our experiments—including OpenEA, DBP15K, OAEI KG Track, DBP1M, and DBpedia–YAGO—in one go from the [shared drive](https://drive.google.com/drive/folders/1L9lPIPIGFr-wA5Hnv_s3uZRDTkBwEAxb?usp=sharing). The shared drive also includes precomputed literal embeddings for OpenEA, DBP15K, and OAEI KG Track. Embeddings for DBP1M and DBpedia–YAGO are not included due to their large size. To run embedding-based literal matching on these two datasets, generate the embeddings using [`src/literal_embedding.py`](../src/literal_embedding.py), as described below.

## Preparing the Data

Extract the downloaded archive into the repository’s `data/` directory. If it contains a top-level `datasets_flora/` folder, move the contents of that folder into `data/`, so that the dataset folders and `emb/` are directly under `data/`.

| Dataset | Dataset directory under `data/` | Embedding directory under `data/emb/` | Precomputed embeddings included |
| --- | --- | --- | --- |
| [OpenEA](https://github.com/nju-websoft/OpenEA) | `OpenEA/D_W_15K_V1/`, `OpenEA/D_W_15K_V2/` | `D_W_15K_V1/`, `D_W_15K_V2/` | Yes |
| [DBP15K](https://github.com/nju-websoft/JAPE) | `DBP15k/fr_en/`, `DBP15k/ja_en/`, `DBP15k/zh_en/` | `fr_en/`, `ja_en/`, `zh_en/` | Yes |
| [OAEI KG Track](https://oaei.ontologymatching.org/2024/knowledgegraph/index.html) | `OAEI/memoryalpha-stexpanded/`, `OAEI/starwars-swtor/` | `memoryalpha-stexpanded/`, `starwars-swtor/` | Yes |
| [DBP1M](https://github.com/ZJU-DAILY/LargeEA) | `DBP1M_with_name/de/`, `DBP1M_with_name/fr/` | `DBP1M/de/`, `DBP1M/fr/` (generate locally) | No |
| DBpedia-YAGO | `DBpedia_YAGO/` | `DBpedia_YAGO/` (generate locally) | No |

Each supplied embedding directory contains `kb1.pkl`, `kb2.pkl`, `kb1.npy`, and `kb2.npy`. Keep all four files together.

## Generating Missing Embeddings

Run embedding generation on a machine with a CUDA-capable GPU and CUDA-enabled PyTorch, after setting up the environment described in the [main README](../README.md). The script automatically uses CUDA when available; otherwise it falls back to CPU, which is much slower for these large datasets. This GPU step is separate from the subsequent main alignment loop, which can reuse the saved embeddings on CPUs.

### DBP1M

Generate embeddings for both the German-English (`de`) and French-English (`fr`) pairs using `sentence-transformers/LaBSE` (multilingual):

Example: 
```bash
python src/literal_embedding.py \
    data/DBP1M_with_name/de/kg1.ttl \
    data/DBP1M_with_name/de/kg2.ttl \
    data/emb/DBP1M/de/ \
    --embedding_model sentence-transformers/LaBSE \
    --batch_size 128  \
    --literal_parser fast
```

### DBpedia-YAGO

Generate embeddings for the English DBpedia and YAGO graphs using `Lihuchen/pearl_small`(monolingual):

```bash
python src/literal_embedding.py \
  data/DBpedia_YAGO/dbpedia_en.ttl \
  data/DBpedia_YAGO/yago_en.ttl \
  data/emb/DBpedia_YAGO/ \
  --embedding_model Lihuchen/pearl_small \
  --batch_size 128 \
  --literal_parser fast
```

The script creates the output directory and writes `kb1.pkl`, `kb2.pkl`, `kb1.npy`, and `kb2.npy`. If GPU memory is insufficient, lower `--batch_size`, for example to `32`.

When running [`src/main.py`](../src/main.py), set `--embedding` to the generated directory, such as `data/emb/DBP1M/de/` or `data/emb/DBpedia_YAGO/`.

## Trying Other Embedding Models

You can experiment with other embedding models that may work better for your datasets by changing `--embedding_model` to a compatible Hugging Face model identifier. The script uses `AutoTokenizer` and `AutoModel`, followed by mean pooling and L2 normalization; models requiring different pooling or special preprocessing may need changes to the script.
