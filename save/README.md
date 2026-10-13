# Save Directory

`save/` stores files generated automatically during FLORA+ runs: alignment results, logs, checkpoints, and preprocessing caches. Input datasets and precomputed literal embeddings belong in [`data/`](../data/README.md).

## Directory Overview

| Directory | Contents | When files are created |
| --- | --- | --- |
| `results/` | Final alignment outputs (`.ttl`). | When a run finishes and writes its `--output` file. (default location) |
| `logs/` | Run progress, timings, and diagnostic messages. | Automatically when the main program starts logging. |
| `checkpoints/` | Saved iteration states for resuming a run. | When `--enable_checkpoint` is enabled and the checkpoint interval is nonzero. (default location)|
| `cache/` | Loaded KGs, compact graph arrays, predicate functionalities, and literal matching results. | Reusable preprocessing caches require `--enable_preprocessing_cache`. |

For example, a run with `--output ../save/results/my_dataset.ttl` can produce:

```text
save/
├── README.md
├── results/
│   └── my_dataset.ttl
├── logs/
│   └── log_my_dataset.txt
├── checkpoints/                  # Optional
│   └── my_dataset/
│       └── checkpoint_iter_0001.pkl
└── cache/
    ├── kb/                      # Cached KG loading
    ├── compact_kg/              # Compact graph arrays
    ├── functionalities/         # Cached predicate functionalities
    └── literal_matching/        # Cached literal similarity scores
```

Not every run creates every directory shown above.

## Results and Logs

Use `--output` to specify the location for the final alignment file. The default location is `save/results/`, but an explicit path can point elsewhere. The output contains Turtle prefix declarations, entity and literal alignments (`owl:sameAs`), and predicate mappings (`rdfs:subPropertyOf`), with alignment scores recorded in comments.

The raw output includes all entity and literal candidates remaining in the final alignment state with scores greater than zero, including low-scoring candidates and multiple possible targets for the same source. For example, the output may contain:

```turtle
@prefix ex: <http://example.org/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
ex:source_A owl:sameAs ex:target_B . # 0.93
ex:source_A owl:sameAs ex:target_C . # 0.41
ex:source_A owl:sameAs ex:target_D . # 0.03
```

These are candidate alignments with different scores, rather than three equally confident matches. Entity and literal output is not restricted to a high-confidence subset or a single best target. Predicate mappings have a separate output condition: only mappings with scores greater than `0.1` are written.
Post-processing is generally needed before evaluation or downstream use.

Logs are written to `save/logs/log_<output_stem>.txt`, where `<output_stem>` is the output filename without its extension. For example, `my_dataset.ttl` produces `log_my_dataset.txt`.

## Caches and Checkpoints

Caches speed up repeated runs on the same dataset; checkpoints resume alignment iterations. A cache alone cannot restore an interrupted iteration loop.

### Preprocessing Caches

Reusable preprocessing caches are disabled by default. Add `--enable_preprocessing_cache` to reuse KG loading, predicate functionalities, and literal matching results on later compatible runs. The program checks cache signatures and recomputes results when a cache is missing or incompatible.

### Checkpoints and Resuming

Checkpoint saving is disabled by default. The following options control it:

| Option | Behavior |
| --- | --- |
| `--enable_checkpoint` | Enable saving alignment state at checkpoint boundaries. |
| `--checkpoint_interval N` | Save every `N` completed iterations; the default is `1`. Set to `0` to disable checkpoint writes. |
| `--checkpoint_dir DIR` | Choose the checkpoint directory.|
| `--resume_checkpoint` | Load the most recently saved compatible checkpoint from that directory. |

Checkpoint files are named `checkpoint_iter_0001.pkl`, `checkpoint_iter_0002.pkl`, and so on; an initialization checkpoint is named as `checkpoint_iter_0000.pkl`.

For example, run the toy dataset with caching and checkpoint saving enabled. From the repository root, first run `cd src`, then:

```bash
python main.py \
  --kg1 ../data/small-test/mini/mini1.ttl \
  --kg2 ../data/small-test/mini/mini2.ttl \
  --string_identity \
  --output ../save/results/mini-test.ttl \
  --enable_preprocessing_cache \
  --enable_checkpoint \
  --checkpoint_interval 1
```

To resume, repeat the same command with `--resume_checkpoint` added:

```bash
python main.py \
  --kg1 ../data/small-test/mini/mini1.ttl \
  --kg2 ../data/small-test/mini/mini2.ttl \
  --string_identity \
  --output ../save/results/mini-test.ttl \
  --enable_preprocessing_cache \
  --enable_checkpoint \
  --checkpoint_interval 1 \
  --resume_checkpoint
```

Keep the input datasets, embeddings, and relevant alignment settings consistent with the saved run. Incompatible checkpoints are skipped; if no compatible checkpoint is found, the program starts a fresh run. Including `--enable_checkpoint` in the resumed command allows it to continue saving checkpoints. Resuming also starts a new log file at the same log path, so copy the previous log first if you need to retain it.

## Retaining and Cleaning Generated Files

Wait until all runs using these files have finished before cleaning them up.

- **Results and logs:** retain the files needed for evaluation, comparison, or experiment records. Deleting them removes those records.
- **Checkpoints:** retain them while you need to resume a run. Deleting them removes the saved iteration state.
- **Caches:** can be removed after runs finish. Later runs will rebuild the required data, which can increase preprocessing time. Retain caches if you want faster repeated runs.
