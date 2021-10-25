# MetaSLU

This project explores the joint learning of multi-task in the few-shot setting, which is the code of the paper 
[Learning to Bridge Metric Spaces: Few-shot Joint Learning of Intent Detection and Slot Filling](https://arxiv.org/abs/2106.07343).

We conduct experiments with two most common tasks in dialog system (**Spoken Language Understanding**): 
 **Intent Detection** and **Slot Filling**. 


## Result

#### FewJoint1.0

- 1-shot

| Models | Intent Acc. | Slot F1 | Joint Acc |
| ------ | ----------- | ------- | --------- |
| SepProto | 66.35 | 27.24 | 10.92 |
| JointProto | 58.52 | 29.49 | 9.64 |
| LD-Proto	|67.70	|27.73	|13.7|
| LD-Proto + TR	|67.63	|34.06	|16.98|
| ConProm (Ours)	|65.26	|33.09	|16.32|
| ConProm+TR (Ours)	|65.73	|37.97	|19.57|
| JointTransfer	|41.83	|26.89	|12.27|


- 5-shot

| Models | Intent Acc. | Slot F1 | Joint Acc |
| ------ | ----------- | ------- | --------- |
|SepProto	|75.64	|36.08	|15.93|
|JointProto	|70.93	|39.47	|14.48|
|LD-Proto	|78.29	|39.88	|22.91|
|LD-Proto + TR	|75.75	|51.62	|27.59|
|ConProm (Ours)	|78.05	|39.4	|24.18|
|ConProm+TR (Ours)	|75.54	|50.28	|28.69|
|JointTransfer	|57.5	|29	|18.81|

#### FewJoint1.1

- 1-shot

| Models | Intent Acc. | Slot F1 | Joint Acc |
| ------ | ----------- | ------- | --------- |
|SepProto	|68.29	|33.16	|15.71|
|JointProto	|59.41	|35.51	|15.81|
|LD-Proto	|69.08	|34.53	|19.44|
|LD-Proto + TR	|67.64	|45.20	|22.69|
|ConProm (Ours)	|66.51	|39.07	|21.09|
|ConProm+TR (Ours)	|75.13	|55.15	|32.54|
|JointTransfer	|45.22	|28.98	|14.79|


- 5-shot

| Models | Intent Acc. | Slot F1 | Joint Acc |
| ------ | ----------- | ------- | --------- |
|SepProto	|76.42	|40.08	|20.40|
|JointProto	|66.94	|40.08	|17.95|
|LD-Proto	|77.70	|41.14	|22.23|
|LD-Proto + TR	|77.01	|53.23	|27.94|
|ConProm (Ours)	|77.93	|43.68	|25.07|
|ConProm+TR (Ours)	|77.07	|57.11	|34.44|
|JointTransfer	|62.08	|33.11	|23.53|

## Get Started

#### Requirement
```bash
python >= 3.6
pytorch >= 0.4.1
pytorch_pretrained_bert >= 0.6.1
allennlp >= 0.8.2
pytorch-nlp
```

#### Step1: Prepare BERT embedding:

- Download the pytorch bert model, or convert tensorflow param by yourself, or click here: [password: atma](https://pan.baidu.com/s/1y9svhPrBpECXum63Ikq-7A)

- Set BERT path in the file `./scripts/smp_ConProm.sh` to your setting:
```bash
bert_base_uncased=/your_dir/chinese_L-12_H-768_A-12/
bert_base_uncased_vocab=/your_dir/chinese_L-12_H-768_A-12/vocab.txt
```

#### Step2: Prepare data
- For snips, FewJoint1.0 and FewJoint1.1, you can get them in `data` here.
- Set test, train, dev data file path in `./scripts/smp_ConProm.sh` to your setting.
> For simplicity, you only need to set the root path for data as follow:
```bash
base_data_dir=/your_dir/your_data_dir/result/SMP_data_slu_1/
```

#### Step3: Train and test the main model

- Build a folder to collect running log
```bash
mkdir result
```
- Execute cross-evaluation script with two params: 
	- [gpu id] 
	- [dataset name]
		- snips or smp

Example for 1-shot Snips:
```bash
source ./scripts/snips_ConProm.sh 0 snips
```
Example for 1-shot Smp:
```bash
source ./scripts/smp_ConProm.sh 0 smp
```

> To run 5-shots experiments, use `./scripts/snips_ConProm_5.sh`

#### Step4: Check your predict results (if `do_predict` is set to True)

- For 1-shot and SMP, you can find the results in `/your_dir/your_data_dir/result/SMP_data_slu_1/smp.spt_s_1.q_s_16.ep_50.cross_id_0/your_model_name_dir`
- For 5-shot and SMP, you can find the results in `/your_dir/your_data_dir/result/SMP_data_slu_5/smp.spt_s_5.q_s_8.ep_50.cross_id_0/your_model_name_dir`
- For 1-shot or 5-shot about snips, you can find the results in `/data/snips` according to your model's name

> You should read the script carefully to get the model's name

## Model for Other Setting
We also provide scripts of four model settings as follows:
- ConProm + TR
- LD-Proto
- LD-Proto + TR
- JointProto
- SepProto (just modify the param `task` as `intent` or `slot_filling`)

> You can find their corresponding scripts in `./scripts/` with the same usage as above.

> After you execute the script, you can view the specific parameters in the result logs.

## Project Architecture

### `Root`
- the project contains three main parts:
	- `models`: the neural network architectures
	- `scripts`: running scripts for cross evaluation
	- `tools`: some tools scrips for get batch score from result files
	- `utils`: auxiliary or tool function files
	- `main.py`: the entry of few-shot models

### `models`
- Main Model
	- Sequence Labeler (`few_shot_seq_labeler.py`): a framework that integrates modules below to perform sequence labeling.
	- Text Classifier (`few_shot_text_classifier.py`): a framework that integrates modules below to perform text classifier.
	- Joint Slot Filling and Intent Detection (`few_shot_slu.py`): a framework that integrates modules below to perform joint slot filling and intent detection.
	- Model with Fine-tune (`normal_slu.py`): a framework that integrates modules below to perform classical supervised model with support set.
- Modules
	- Embedder Module (`context_embedder_base.py`): modules that provide embeddings.
	- Emission Module (`emission_scorer_base.py`): modules that compute emission scores.
	- Transition Module (`transition_scorer.py`): modules that compute transition scores.
	- Similarity Module (`similarity_scorer_base.py`): modules that compute similarities for metric learning based emission scorer.
	- Output Module (`seq_labeler.py`, `conditional_random_field.py`): output layer with normal mlp or crf.
	- Scale Module (`scale_controller.py`): a toolkit for re-scale and normalize logits.

### `utils`
- utils contains assistance modules for:
	- data processing (`data_loader.py`, `preprocessor.py`),
	- constructing model architecture (`model_helper.py`),
	- controlling training process (`trainer.py`),
	- controlling testing process (`tester.py`),
	- controllable parameters definition (`opt.py`),
	- device definition (`device_helper`)
	- config (`config.py`).


### `tools`
- get batch scores from result files(`count_score.py`)
	- usage
	```bash
	python count_score.py [snip/smp] [result_file_path - without $cross_id.m_seed_$seed]
	```
	- example
	```bash
	python count_score.py smp ../data/smp_release_new/smp.spt_s_1.q_s_16.ep_50.cross_id_0/ds_SepProto.intent.bert.ga_4.bs_4.ep_4.lr_0.00001.up_lr_0.001.dec_sms.DATA.smp.shots_1.cross_id_
	```
- get semantic acc score for SepProto(`count_semantic_acc_for_sep_proto.py`)
	- usage
	```bash
	python count_semantic_acc_for_sep_proto.py [snip/smp] [result_file_path - without $cross_id.m_seed_$seed]
	```
	- example
	```bash
	python count_semantic_acc_for_sep_proto.py smp ../data/smp_release_new/smp.spt_s_1.q_s_16.ep_50.cross_id_0/ds_SepProto.intent.bert.ga_4.bs_4.ep_4.lr_0.00001.up_lr_0.001.dec_sms.DATA.smp.shots_1.cross_id_
	```
- get slot acc from result files(`cal_sentence_level_slot_acc.py`)
	- usage
	```bash
	python cal_sentence_level_slot_acc.py [snip/smp] [result_file_path - without $cross_id.m_seed_$seed]
	```
	- example
	```bash
	python cal_sentence_level_slot_acc smp ../data/smp_release_new/smp.spt_s_1.q_s_16.ep_50.cross_id_0/ds_SepProto.intent.bert.ga_4.bs_4.ep_4.lr_0.00001.up_lr_0.001.dec_sms.DATA.smp.shots_1.cross_id_
	```
- format batch scores as (`format_res.py`)
	- usage
		- copy the result scores get from `count_score.py` in `test.txt`
		- run follow scripts:
		```bash
		python format_res
		```
		- get formatted result scores in `res.txt` 


## Important Parameters (you can get more detailed information in `/utils/opt.py`)

#### Task: 
- `intent` for Intent Detection
- `slot_filling` for Slot Filling
- `slu` for Joint Slot Filling and Intent Detection

#### Few-Shot Models

- `slu_model_type`: few-shot model type
- `proto_merge_type`: ProtoMerge methods
- `split_metric`: the type of SplitMetric module
- `proto_update`: the method for merging ProtoMerge methods 
- `slu_regular`: the type of regular loss(use metric learning method - contrastive learning)
- `margin`: the margin in Metric Learning
- `with_homo`: add contrastive loss between prototypes in sampe task
- `seed`: seed
- `epoch`: epoch of training
- `decoder`: the type of decoder in Slot Filling
	- `sms`: simple sequence decoder
	- `rule`: use simple transition rule decoder


