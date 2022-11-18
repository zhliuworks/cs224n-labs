# Experiment results

```shell
mkdir params preds
```

```python
import pandas as pd
pd.Series(list(map(lambda x: len(x), open('wiki.txt').read().split('\n')))).describe()
```


## Train from scratch

### train
```shell
python src/run.py finetune vanilla wiki.txt \
--writing_params_path params/vanilla.scratch.pt \
--finetune_corpus_path birth_places_train.tsv
```

### evaluate
```shell
python src/run.py evaluate vanilla wiki.txt \
--reading_params_path params/vanilla.scratch.pt \
--eval_corpus_path birth_dev.tsv \
--outputs_path preds/vanilla.scratch.dev.txt
```

> [ACC] 1.2%

> for "London" baseline, [ACC] 5.0%

### inference
```shell
python src/run.py evaluate vanilla wiki.txt \
--reading_params_path params/vanilla.scratch.pt \
--eval_corpus_path birth_test_inputs.tsv \
--outputs_path preds/vanilla.scratch.test.txt
```


## Pretrain + finetune

### pretrain
```shell
python src/run.py pretrain vanilla wiki.txt \
--writing_params_path params/vanilla.pretrain.pt
```

### finetune
```shell
python src/run.py finetune vanilla wiki.txt \
--reading_params_path params/vanilla.pretrain.pt \
--writing_params_path params/vanilla.finetune.pt \
--finetune_corpus_path birth_places_train.tsv
```

### evaluate
```shell
python src/run.py evaluate vanilla wiki.txt \
--reading_params_path params/vanilla.finetune.pt \
--eval_corpus_path birth_dev.tsv \
--outputs_path preds/vanilla.finetune.dev.txt
```

> [ACC] 26.4%

### inference
```shell
python src/run.py evaluate vanilla wiki.txt \
--reading_params_path params/vanilla.finetune.pt \
--eval_corpus_path birth_test_inputs.tsv \
--outputs_path preds/vanilla.finetune.test.txt
```


## Synthesizer

### pretrain
```shell
python src/run.py pretrain synthesizer wiki.txt \
--writing_params_path params/synthesizer.pretrain.pt
```

### finetune
```shell
python src/run.py finetune synthesizer wiki.txt \
--reading_params_path params/synthesizer.pretrain.pt \
--writing_params_path params/synthesizer.finetune.pt \
--finetune_corpus_path birth_places_train.tsv
```

### evaluate
```shell
python src/run.py evaluate synthesizer wiki.txt \
--reading_params_path params/synthesizer.finetune.pt \
--eval_corpus_path birth_dev.tsv \
--outputs_path preds/synthesizer.finetune.dev.txt
```

> [ACC] 9.6%

### inference
```shell
python src/run.py evaluate synthesizer wiki.txt \
--reading_params_path params/synthesizer.finetune.pt \
--eval_corpus_path birth_test_inputs.tsv \
--outputs_path preds/synthesizer.finetune.test.txt
```
