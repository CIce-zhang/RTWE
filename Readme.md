# Word Sense Disambiguation by Refining Target Word Embedding

For the base model, run the following code. Note that the WSD evaluation framework should be downloaded, the argument '--data-path' is the path to WSD evaluation framework.
```bash
python -u biencoder-context.py --context_mode nonselect --encoder-name roberta-base --train_mode roberta-base --gloss_mode non --same --sec_wsd --rtwe --data-path ../WSD_Evaluation_Framework
```

For the large model, run the following code.
```bash
python -u biencoder-context.py --context_mode nonselect --encoder-name roberta-large --train_mode roberta-large --lr 1e-6 --gloss-bsz 150 --gloss_mode non --same --sec_wsd --rtwe --data-path ../WSD_Evaluation_Framework
```

For the large model that trains with more training data (SemCor+WNGE), run the following code.
```bash
python -u biencoder-context.py --context_mode nonselect --encoder-name roberta-large --train_mode roberta-large --lr 1e-6 --gloss-bsz 150 --gloss_mode non --same --sec_wsd --rtwe --train_data semcor-wngt --data-path ../WSD_Evaluation_Framework
```
