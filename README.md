# LeanCode
This repo provides the code for reproducing the experiments in LeanCode. LeanCode is a novel program simplification approach that utilize code contexts and attention scores for representing the importance levels of tokens.
## Requirements
- [python3](https://www.python.org/downloads/)
- [PyTorch](https://pytorch.org/get-started/locally/)
## QuickStart
### Prepare data
We used two datasets from [CodeBERT](https://arxiv.org/pdf/2002.08155): code search and code summarization datasets. These are the extensions of [CodeSearchNet](https://github.com/github/CodeSearchNet).
You can download the code search dataset from [this website](https://drive.google.com/uc?id=1xgSR34XO8xXZg4cZScDYj2eGerBE9iGo), and download the code summarization dataset from [this website](https://drive.google.com/uc?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h) Or use the following command.

```
mkdir data data/codesearch data/code2nl
cd data/codesearch
gdown https://drive.google.com/uc?id=1xgSR34XO8xXZg4cZScDYj2eGerBE9iGo
unzip codesearch_data.zip
rm  codesearch_data.zip
cd ../../codesearch
python process_data.py

cd ../data/code2nl
gdown https://drive.google.com/uc?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h
unzip Cleaned_CodeSearchNet.zip
rm Cleaned_CodeSearchNet.zip
cd ../..
```

### Training
you can use the following cmd to train the model for code search and code summarization on CodeT5 and CodeBERT:

code search on CodeBERT:
```
python3 codesearch/run_classifier.py --model_type roberta --tokenizer_name microsoft/codebert-base --model_name_or_path microsoft/codebert-base --task_name codesearch --do_train --do_eval --prune_strategy None --output_dir ./models/codesearch/codebert/base --data_dir ./data/codesearch/train_valid/java --train_file train.txt --dev_file valid.txt --max_seq_length 512 --per_gpu_train_batch_size 64 --per_gpu_eval_batch_size 64 --learning_rate 1e-5 --num_train_epochs 4  --lang java --gradient_accumulation_steps 1 --overwrite_output_dir
```
code search on CodeT5:
```
python3 codesearch/run_classifier.py --model_type codet5 --tokenizer_name Salesforce/codet5-base --model_name_or_path Salesforce/codet5-base --task_name codesearch --do_train --do_eval --prune_strategy None --output_dir ./models/codesearch/codet5/base --data_dir ./data/codesearch/train_valid/java --train_file train.txt --dev_file valid.txt --max_seq_length 512 --per_gpu_train_batch_size 64 --per_gpu_eval_batch_size 64 --learning_rate 1e-5 --num_train_epochs 4  --lang java --gradient_accumulation_steps 1 --overwrite_output_dir
```
code summarization on CodeBERT:
```
python code2nl/CodeBERT/run.py --model_type roberta --tokenizer_name microsoft/codebert-base --model_name_or_path microsoft/codebert-base --do_train --do_eval --prune_strategy None --train_filename ./data/code2nl/CodeSearchNet/java/train.jsonl --dev_filename ./data/code2nl/CodeSearchNet/java/valid.jsonl --output_dir ./models/code2nl/codebert/base --max_source_length 512 --max_target_length 128 --beam_size 10 --train_batch_size 128 --eval_batch_size 128 --learning_rate 5e-5 --num_train_epochs 15
```
code summarization on CodeT5:
```
python code2nl/CodeT5/run_gen.py --model_type codet5 --task summarize --sub_task java --tokenizer_name Salesforce/codet5-base --model_name_or_path Salesforce/codet5-base --do_train --do_eval --do_eval_bleu  --prune_strategy None --data_num -1 --num_train_epochs 8 --warmup_steps 1000 --learning_rate 5e-5 --patience 2 --data_dir ./data/code2nl/CodeSearchNet/java --cache_path ./models/code2nl/codet5/base/cache_data --output_dir ./models/code2nl/codet5/base --save_last_checkpoints --always_save_model --res_dir ./models/code2nl/codet5/base/prediction --res_fn ./models/code2nl/codet5/base/result.txt --train_batch_size 96 --eval_batch_size 96 --max_source_length 512 --max_target_length 128 --summary_dir ./models/code2nl/codet5/base/tensorboard
```
### Prepare pruned test data by slimcode and dietcode
Because of the algorithm of SlimCode need to remove all the comments in the code and remove the code that can't be converted to AST after removing the comments , so ,we download the test data from [SlimCode](https://github.com/gksajy/slimcode?tab=readme-ov-file)

Go to `slimcode` folder , run SlimCode.java ,you can get the pruned data of code search and summarization by SlimCode .

Go to `codesearch`,with the help of prune_dietcode.py , you can get the pruned data of code search by DietCode and LeanCode with the removal algorithm of DietCode

run codesearch/process_pruned_data.py

### Reproduce the experimental results
We conducted code simplification experiments on two tasks, `code search` and `code summarization`, using two models: `CodeBERT` and `CodeT5`. These experiments involved three code simplification strategies—`LeanCode`,`SlimCode` , `DietCode`, and `LeanCode with the removal algorithm of DietCode` and five code simplification ratios: `10%`, `20%`, `30%`, `40%`, and `50%`.  
  
You can use `gen_cmd.py` in `utils` floder to get the inference cmd to run a broad set of experiments by simply passing the `task_type`, `model_type`, `prune_strategy` and `prune_ratio` arguments. 
Below is a table listing each parameter along with its description and available options:

| Parameter          | Description                                           | Options                                                         |
|--------------------|-------------------------------------------------------|-----------------------------------------------------------------|
| **task_type**      | Specifies the type of task to perform.                | `codesearch`, `code2nl`                                         |
| **model_type**     | Determines the model used for the task.               | `codebert`, `codet5`                                            |
| **prune_strategy** | Defines the pruning strategy to enhance model performance. | `leancode`,`slimcode`, `dietcode`, `leancode_d`, `None`                  |
| **prune_ratio**    | Specifies the percentage of the model to prune.       | `10`, `20`, `30`, `40`, `50`                                    |

#### Usage Examples
for example , If you want to perform a code search task using the CodeBERT model with a LeanCode strategy at a 40% pruning ratio. execute the command as shown below ，then you can obtain the corresponding infrence command, which then allow you to conduct the respective experiments

```
python utils/gen_cmd.py --task_type codesearch --model_type codebert --prune_strategy leancode --prune_ratio 40
```
If you need to conduct an experiment on CodeBERT for code search without pruning (base expirement), you can use the following commands to obtain the necessary command for inference.
```
python utils/gen_cmd.py --task_type codesearch --model_type codebert --prune_strategy None
```
### Generate the weight_dic of LeanCode
If you want to generate the weight dictionaries by yourself , add --gen_weight when you train the base model and the gen_weight_dic.py in the `utils` folder will help you generate the weight_dic
### Other works
- the `analyse_attention` folder can help you analyse self_attention , self attention of cls token and encoder-decoder attention with the example of BubbleSort.  
- the `alalyse_weight.py` in `utils` folder can help you get Statistics of cls attention scores and encoder-decoder attention scores based on the training dataset .
