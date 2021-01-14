# Discern: Discourse-Aware Entailment Reasoning Network for Conversational Machine Reading (EMNLP 2020)

This repository is the implementation of our EMNLP 2020 Paper [Discern: Discourse-Aware Entailment Reasoning Network for Conversational Machine Reading](https://arxiv.org/abs/2010.01838).

`Discern` achieves new state-of-the-art results on [ShARC conversational machine reading benchmark](https://sharc-data.github.io/leaderboard.html) (May 2020).

If you have any question, please open an issue or contact yifangao95@gmail.com

## Reference

If you find our code useful, please cite our papers as follows:

```bibtex
@article{gao-etal-2020-discern,
  title={Discern: Discourse-Aware Entailment Reasoning Network for Conversational Machine Reading},
  author={Yifan Gao and Chien-Sheng Wu and Jingjing Li and Shafiq Joty and Steven C.H. Hoi, Caiming Xiong and Irwin King and Michael R. Lyu},
  journal={EMNLP},
  year={2020},
}
```
```bibtex
@article{gao-etal-2020-explicit,
  title={Explicit Memory Tracker with Coarse-to-Fine Reasoning for Conversational Machine Reading},
  author={Yifan Gao and Chien-Sheng Wu and Shafiq R. Joty and Caiming Xiong and Richard Socher and Irwin King and Michael R. Lyu and Steven C. H. Hoi},
  journal={ACL},
  year={2020},
}
```


## Requirements
> Discourse segmentation environment (`PYT_SEGBOT`)

```bash
conda create -n segbot python=3.6
conda install pytorch==0.4.1 -c pytorch
conda install nltk==3.4.5 numpy==1.18.1 pycparser==2.20 six==1.14.0 tqdm==4.44.1
```

> Main environment (`PYT_DISCERN`)

```bash
conda create -n discern python=3.6
conda install pytorch==1.0.1 cudatoolkit=10.0 -c pytorch
conda install spacy==2.0.16 scikit-learn
python -m spacy download en_core_web_lg && python -m spacy download en_core_web_md
pip install editdistance==0.5.2 transformers==2.8.0
```

> UniLM question generation environment (`PYT_QG`)

```bash
# create conda environment
conda create -n qg python=3.6
conda install pytorch==1.1 cudatoolkit=10.0 -c pytorch
conda install spacy==2.0.16 scikit-learn
python -m spacy download en_core_web_lg && python -m spacy download en_core_web_md
pip install editdistance==0.5.2

# install apex
git clone -q https://github.com/NVIDIA/apex.git
cd apex
git reset --hard 1603407bf49c7fc3da74fceb6a6c7b47fece2ef8
python setup.py install --cuda_ext --cpp_ext
cd ..

# setup unilm
cd qg
pip install --editable .
```

> Download ShARC data
```bash
mkdir data
cd data
wget https://sharc-data.github.io/data/sharc1-official.zip -O sharc_raw.zip
unzip sharc_raw.zip
mv sharc1-official/ sharc_raw
```

> Download RoBERTa, UniLM
```bash
mkdir pretrained_models
# RoBERTa
mkdir pretrained_models/roberta_base
wget --quiet https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-config.json -O pretrained_models/roberta_base/config.json
wget --quiet https://cdn.huggingface.co/roberta-base-merges.txt -O pretrained_models/roberta_base/merges.txt
wget --quiet https://cdn.huggingface.co/roberta-base-pytorch_model.bin -O pretrained_models/roberta_base/pytorch_model.bin
wget --quiet https://cdn.huggingface.co/roberta-base-vocab.json -O pretrained_models/roberta_base/vocab.json
# UniLM & BERT
mkdir pretrained_models/unilm
wget --quiet https://unilm.blob.core.windows.net/ckpt/unilm1-large-cased.bin -O pretrained_models/unilm/unilmv1-large-cased.bin
wget --quiet https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt -O pretrained_models/unilm/bert-large-cased-vocab.txt
wget --quiet https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz -O pretrained_models/unilm/bert-large-cased.tar.gz
cd pretrained_models/unilm
tar -zxvf bert-large-cased.tar.gz
rm bert-large-cased.tar.gz
```
You can also download our pretrained models and our dev set predictions:
- Decision Making Model: [decision.pt](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155102332_link_cuhk_edu_hk/ESrX-zVkGjZBsHofDoChXIYB1UVlgld3jJZyeJcZApemCQ?e=ggQy1w)
- Span Extraction Model: [span.pt](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155102332_link_cuhk_edu_hk/EZ7e3LDwYZlHtg917fTsK2gBUp7HD52wzE65mKSx4FY5uQ?e=kVXjyC)
- Question Generation Model: [unilmqg.bin](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155102332_link_cuhk_edu_hk/ER4GRoby0ORGjIYUKLo-Tc4Bq-G7De5qElmJh_Rt_EtGqQ?e=8Yeuix)
> We would now set up our directories like this:

```
.
└── model
    └── ...
└── segedu
    └── ...
└── unilmqg
    └── ...
└── README.md
└── data
    └── ...
└── pretrained_models
    └── unilm
        └── ...
        └── unilmqg.bin
    └── roberta_base
        └── ...
    └── decision.pt
    └── span.pt
```

## Discourse Segmentation of Rules (Section 2.1)

We use [SegBot](http://138.197.118.157:8000/segbot/) and [their implementation](https://www.dropbox.com/sh/tsr4ixfaosk2ecf/AACvXU6gbZfGLatPXDrzNcXCa?dl=0) to segment rules in the ShARC regulation snippets.

```shell
cd segedu
PYT_SEGBOT preprocess_discourse_segment.py
PYT_SEGBOT sharc_discourse_segmentation.py
```

`data/train_snippet_parsed.json` and `data/dev_snippet_parsed.json` are parsed rules.

## Fix Questions in ShARC

We find in some cases, there are some extra/missing spaces in ShARC questions. Here we fix them by merging these questions:

```shell
PYT_DISCERN fix_questions.py
```

## Decision Making (Section 2.2)

> preprocess: prepare inputs for RoBERTa, generate labels for entailment supervision

```shell
PYT_DISCERN preprocess_decision.py
```

> training

```shell
PYT_DISCERN -u train_sharc.py \
--train_batch=16 \
--gradient_accumulation_steps=2 \
--epoch=5 \
--seed=323 \
--learning_rate=5e-5 \
--loss_entail_weight=3.0 \
--dsave="out/{}" \
--model=decision \
--early_stop=dev_0a_combined \
--data=./data/ \
--data_type=decision_roberta_base \
--prefix=train_decision \
--trans_layer=2 \
--eval_every_steps=300  # 516
```

> inference

Here we can directly do interence using our trained model `decision.pt`. You can also replace it with your own models by setting `--resume=/path/to/your/trained/models`.

```shell
PYT_DISCERN train_sharc.py \
--dsave="./out/{}" \
--model=decision \
--data=./data/ \
--data_type=decision_roberta_base \
--prefix=inference_decision \
--resume=./pretrained_models/decision.pt \
--trans_layer=2 \
--test
```

The prediction file is saved at './out/inference_decision/dev.preds.json'.
Our model achieves the following performance on the development set using our pre-trained model `decision.pt`:

| Micro Acc. | Macro Acc. |
|:----------:|:----------:|
|    74.85   |    79.79   |


## Follow-up Question Generation (Section 2.3)

For the follow-up question generation task, we firstly use a span-extraction model to extract the underspecified span within the rule text, then use UniLM to rephrase the span into a well-formed question.

### Span Extraction

> preprocess span extraction

```
PYT_DISCERN preprocess_span.py
```

> training

```shell
PYT_DISCERN -u train_sharc.py \
--train_batch=16 \
--gradient_accumulation_steps=2 \
--epoch=5 \
--seed=115 \
--learning_rate=5e-5 \
--dsave="out/{}" \
--model=span \
--early_stop=dev_0_combined \
--data=./data/ \
--data_type=span_roberta_base \
--prefix=train_span \
--eval_every_steps=100
```

> inference

```shell
PYT_DISCERN -u train_sharc.py \
--dsave="out/{}" \
--model=span \
--data=./data/ \
--data_type=span_roberta_base \
--prefix=inference_span \
--resume=./pretrained_models/span.pt \
--test
```

Our trained model 'span.pt' achieves the following intermediate results:

| BLEU 1 | BLEU 4 | Span_F1 |
|--------|--------|---------|
| 50.89  | 44.0   | 62.59   |


### UniLM Question Generation

We follow [Explicit Memory Tracker](https://github.com/Yifan-Gao/explicit_memory_tracker) for question generation.
Here we take their trained model and do inference only.
Please refer to the [Explicit Memory Tracker](https://github.com/Yifan-Gao/explicit_memory_tracker) repo for training details.

The UniLM Question Generation model reads the predicted span from the span extraction model, and rephrases it into the question.

```shell
PYT_QG -u qg.py \
--fin=./data/sharc_raw/json/sharc_dev.json \
--fpred=./out/inference_span \  # directory of span prediction
--model_recover_path=/absolute/path/to/pretrained_models/qg.bin \
--cache_path=/absolute/path/to/pretrain_models/unilm/
```

Oracle question generation evaluation results of our released model `unilmqg.bin` (dev. set):

| BLEU 1 | BLEU 2 | BLEU 3 | BLEU 4 |
|--------|--------|--------|--------|
| 65.73  | 59.43  | 55.43  | 52.43  |


## End-to-End Evaluation (TODO)



