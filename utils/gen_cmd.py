import os
import argparse


def generate_command_codesearch(task_type, model_type, pruning_strategy, pruning_ratio):
    if pruning_strategy=='None':
        model_dir = f"./models/{task_type}/{model_type}/base"
        data_path = f"./data/{task_type}/test/java"
        ratio=-1
    else:
        model_dir = f"./models/{task_type}/{model_type}/base/{pruning_strategy}/{pruning_ratio}"
        ratios={10:0.9,20:0.8,30:0.7,40:0.6,50:0.5}
        ratio=ratios[pruning_ratio]
        if pruning_strategy=='leancode':
            data_path = f"./data/{task_type}/test/java"
        elif pruning_strategy=='slimcode':
            data_path = f"./data/{task_type}/{pruning_strategy}/{model_type}/{pruning_ratio}/test"
        else:
            data_path = f"./data/{task_type}/{pruning_strategy}/{pruning_ratio}/test"

    if model_type=='codebert':
        model_type_='roberta'
        model_path='microsoft/codebert-base'
    else:
        model_type_='codet5'
        model_path = 'Salesforce/codet5-base'
    test_cmd = f"python3 {task_type}/run_classifier.py --model_type {model_type_} --tokenizer_name {model_path} --model_name_or_path {model_path} --task_name codesearch --do_predict --prune_strategy {pruning_strategy} --output_dir {model_dir} --pred_model_dir ./models/{task_type}/{model_type}/base/checkpoint-best --test_result_dir {model_dir}/0_batch_result.txt --data_dir {data_path} --test_file batch_0.txt  --per_gpu_eval_batch_size 50 --learning_rate 1e-5 --ratio {ratio}"
    # test_cmd = f"python3 {task_type}/mrr.py --test_dir {model_dir}/0_batch_result.txt"

    return test_cmd

def generate_command_code2nl_codebert(task_type, model_type, pruning_strategy, pruning_ratio):
    if pruning_strategy=='None':
        model_dir = f"./models/{task_type}/{model_type}/base"
        data_path=f"./data/{task_type}/CodeSearchNet/java/test.jsonl"
        ratio=-1
    else:
        model_dir = f"./models/{task_type}/{model_type}/base/{pruning_strategy}/{pruning_ratio}"
        ratios={10:0.9,20:0.8,30:0.7,40:0.6,50:0.5}
        ratio=ratios[pruning_ratio]
        if pruning_strategy=='slimcode':
            data_path = f"./data/{task_type}/CodeSearchNet/java/slimcode/{model_type}/10/test.txt"
        else :
            data_path = f"./data/{task_type}/CodeSearchNet/java/test.jsonl"
    model_type_ = 'roberta'
    model_path = 'microsoft/codebert-base'

    test_cmd = f"python code2nl/CodeBERT/run.py --model_type {model_type_} --tokenizer_name {model_path} --model_name_or_path {model_path} --do_test --prune_strategy {pruning_strategy} --test_filename {data_path} --output_dir {model_dir}  --beam_size 10 --load_model_path ./models/{task_type}/{model_type}/base/checkpoint-best-bleu/pytorch_model.bin --eval_batch_size 32 --ratio {ratio}"

    return test_cmd

def generate_command_code2nl_codet5(task_type, model_type, pruning_strategy, pruning_ratio):
    ratio = -1
    if pruning_strategy=='None':
        model_dir = f"./models/{task_type}/{model_type}/base"
        data_path=f"./data/{task_type}/CodeSearchNet/java"

    else:
        model_dir = f"./models/{task_type}/{model_type}/base/{pruning_strategy}/{pruning_ratio}"
        ratios={10:0.9,20:0.8,30:0.7,40:0.6,50:0.5}
        ratio=ratios[pruning_ratio]
        if pruning_strategy=='slimcode':
            data_path = f"./data/{task_type}/CodeSearchNet/java/slimcode/{model_type}/10"
        else :
            data_path = f"./data/{task_type}/CodeSearchNet/java"
    model_path = 'Salesforce/codet5-base'
    test_cmd = f"python code2nl/CodeT5/run_gen.py --model_type codet5 --task summarize --sub_task java --tokenizer_name {model_path} --model_name_or_path {model_path} --do_test --prune_strategy {pruning_strategy} --data_num -1 --num_train_epochs 0 --warmup_steps 1000 --learning_rate 5e-5 --patience 2 --data_dir {data_path} --cache_path {model_dir}/cache_data --output_dir {model_dir} --save_last_checkpoints --always_save_model --res_dir {model_dir}/prediction --res_fn {model_dir}/result.txt --train_batch_size 96 --eval_batch_size 32  --max_target_length 128 --summary_dir {model_dir}/tensorboard --ratio {ratio}"
    return test_cmd


def main():
    parser = argparse.ArgumentParser(description="Generate command based on parameters.")

    parser.add_argument("--task_type", type=str, help="Type of the task",choices=['codesearch', 'code2nl'])
    parser.add_argument("--model_type", type=str, help="Type of the model",choices=['codebert', 'codet5'])
    parser.add_argument("--prune_strategy", type=str, help="Pruning strategy",choices=['leancode', 'dietcode', 'leancode_d','slimcode','None'])
    parser.add_argument("--prune_ratio", type=int, help="Pruning ratio",choices=[10,20,30,40,50])

    args = parser.parse_args()

    if args.task_type=='codesearch':
        test_cmd = generate_command_codesearch(args.task_type, args.model_type, args.prune_strategy, args.prune_ratio)
        print('============================Test cmd==========================')
        print(test_cmd)
    elif args.model_type=="codebert":
        assert args.task_type=='code2nl'
        test_cmd=generate_command_code2nl_codebert(args.task_type, args.model_type, args.prune_strategy, args.prune_ratio)
        print('============================Testing cmd==========================')
        print(test_cmd)
    else:
        assert args.task_type=='code2nl'
        assert args.model_type=='codet5'
        test_cmd=generate_command_code2nl_codet5(args.task_type, args.model_type, args.prune_strategy, args.prune_ratio)
        print('============================Training and test cmd==========================')
        print(test_cmd)


if __name__ == "__main__":
    main()
