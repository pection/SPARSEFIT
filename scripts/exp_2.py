import itertools
import ast
import os
import argparse
import glob
import pandas as pd
import numpy as np
import time 
import subprocess

# ===========> Code for collecting results from 60 runs 
def std(x):
    return np.std(x)

def collect_results(args):
    all_experiment_dirs = glob.glob(f'{args.exp_root}/*/*')
    print('These are all the experiment directories')
    print(all_experiment_dirs)
    
    results = []
    for exp_dir in all_experiment_dirs:
        # if 't5-3b' in exp_dir and 't5-base' in exp_dir:
        #     print('Skipping this directory', str(exp_dir), ' because it has t5-3b or t5-base in it')
        #     continue
        if not os.path.isdir(exp_dir):
            continue

        log_path = os.path.join(exp_dir, 'logger.log') #path to log file
        metrics = ['dev_acc', 'dev_bertscore_correct_normalized'] #metrics being monitored
        metrics_dict = {} #stores the monitored metrics for each experiment
        #check log file for the current experiment, and extract the monitored metrics
        with open(log_path) as f:
            for line in f:
                for metric in metrics: 
                    if (metric in line and f"{metric}_correct_pred" not in line) or (metric in line and metric.replace("_correct_pred","") not in line):
                        score = line.strip().split()[-1]
                        metrics_dict[metric] = float(score)

        config_path = os.path.join(exp_dir, 'commandline_args.txt') #path to file storing experiment configurations
        #extract configs (to be used in dataframe) from config file
        with open(config_path) as f:
            lines = f.readlines()
            lines = [l for l in lines[5:]]
            configs = ''.join(lines).replace('\n', '').split('--')
            filtered_configs = {}
            for metric in metrics: 
                filtered_configs[metric] = -1 
            configs_to_keep = ['task_name', 'model_type', 'n_shots', 'seed', 'io_format'] 
            for config in configs:
                for config_name in configs_to_keep:
                    if config.startswith(config_name):
                        config = config.replace(config_name, '')
                        config = config.replace('../checkpoints/', '')  
                        filtered_configs[config_name] = config
                        break
        filtered_configs.update(metrics_dict)
        if filtered_configs['dev_acc'] == -1:
            print (f"There is a failed run. Path to its log file: {log_path}.\n \
                   Remove this serailization directiory from `all_experiment_dirs` if not needed or\
                   re-run this experiment.")
            print(filtered_configs.values())
        else:
            results.append(filtered_configs)

    print('The number of experiments that have metrics calculated is', str(len(results)), ' out of ', str(len(all_experiment_dirs)), 'experiments.')
    df = pd.DataFrame.from_records(results) 
    try:
        assert len(df) % len(seeds_fewshot) == 0 
    except AssertionError: 
        for model in set(df['model_type'].tolist()):
            df_sub = df[df['model_type']==model]
            for io_format in set(df_sub['io_format'].tolist()):
                if len (df_sub[df_sub['io_format']==io_format]) != len(seeds_fewshot):
                    seeds = [x for x in seeds_fewshot if str(x) not in df_sub[df_sub['io_format']==io_format]["seed"].tolist()]
                    print(f"The following seeds failed for model {model} and format {io_format}:")
                    print (seeds)
                    print (f"Repeat experiments for those seeds and collect results again")
    
    #make results directory
    output_file = 'out'
    if not os.path.exists(output_file):
        os.mkdir(output_file)

    columns_to_convert = ['seed', 'n_shots']

    df[columns_to_convert] = df[columns_to_convert].astype('int64')
    
    print(df.to_csv('out/results_all.csv', index=True))

    df_avg_seed = df.groupby(['task_name', 'model_type', 'io_format', 'n_shots']).mean() #key error would occur when the results that was converted to df was empty
    print(df_avg_seed.to_csv('out/results.csv', index=True))

    df_avg_seed_with_std = df.groupby(['task_name', 'model_type', 'io_format', 'n_shots']).agg(['mean', std]) 
    print(df_avg_seed_with_std.to_csv('out/results_with_std.csv', index=True))


# ===========> Code for running models with 60 seeds; eval on dev sets will be done jointly with training and results recorded in logger.log that we will use to collect results 
seeds_fewshot = [7004, 3639, 6290, 9428, 7056, 4864, 4273, 7632, 2689, 8219, 4523, 2175, 7356, 8975, 51, 4199, 4182, 1331, 2796, 6341, 7009, 1111, 1967, 1319, 741, 7740, 1335, 9991, 6924, 6595, 5358, 2638, 6227, 8384, 2769, 9933, 6339, 3112, 1349, 8483, 2348, 834, 6895, 4823, 2913, 9962, 178, 2147, 8160, 1936, 4512, 2051, 4779, 2498, 176, 9599, 1181, 5320, 588, 4791]
seeds_fewshot = [7004]
experiments = {}
# You can add your own values as a new key and run those experiments with `---experiment_id <your_experiment_key>`
experiments['t5_unifiedqa_fewshot'] = { # Values are lists because you can run experiments with different values sequentially 
                                        'seed_vals': seeds_fewshot, 
                                        'dataset_vals': None,
                                        'model_vals': None, 
                                        'early_stopping_patience_vals': [1], 
                                        'max_steps_vals': [300], 
                                        'epochs_vals': [2],  # will be ignored because of `max_steps`
                                        'warmup_steps_vals': [0],
                                        'eval_steps_vals': [300], 
                                        'fewshot_eval_size': [350],
                                        'explanation_sep_vals': ['" because "'],
                                        'tokenizer_vals': None,
                                        'per_device_train_batch_size_vals': [2], 
                                        'learning_rate_vals' : [3e-5],
                                    }


# Assuming you have args.n_gpus only for this project and no queing system, the following queues jobs on the available gpus for you 
# this is from https://stackoverflow.com/questions/53422761/distributing-jobs-evenly-across-multiple-gpus-with-multiprocessing-pool
# from multiprocessing import Pool, current_process, Queue
# NUM_WORKERS = None  # the number of GPUs
# queue = Queue()

def foo(cmd, gpu_list):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_list))
    print(f'Executing command on GPUs {gpu_list}: {cmd}')
    completed = subprocess.call(cmd, shell=True)
    if completed == 0:
        print(f'Command completed successfully on GPUs {gpu_list}')
    else:
        print(f'Command failed with exit code {completed} on GPUs {gpu_list}')

def run_exp(args):
    if not os.path.isdir(args.exp_root):
        os.mkdir(args.exp_root)

    experiment = experiments[args.experiment_id]
    seed_vals = experiment['seed_vals']
    dataset_vals = experiment['dataset_vals']
    model_vals = experiment['model_vals']
    early_stopping_patience_vals = experiment['early_stopping_patience_vals']
    max_steps_vals = experiment['max_steps_vals']
    epochs_vals = experiment['epochs_vals']
    warmup_steps_vals = experiment['warmup_steps_vals']
    learning_rate_vals = experiment['learning_rate_vals']
    per_device_train_batch_size_vals = experiment['per_device_train_batch_size_vals']
    eval_steps_vals = experiment['eval_steps_vals']
    fewshot_eval_size_vals = experiment['fewshot_eval_size']
    explanation_sep_vals = experiment['explanation_sep_vals']
    tokenizer_vals = experiment['tokenizer_vals']

    # You can uncomment lines to try a specific IO format
    # Formats that we did not comment are those that gave us the best results reported in the paper
    format_dict = {'esnli': [
                             #'t5_fewshot_infilling', 
                             #'t5_fewshot_infilling_more_natural', 
                             'standard', 
                             #'squad_endswith_what', 
                             #'squad_nli_mix_endswith_what', 
                             #'squad', 
                             #'squad_nli_mix',
                             #'unifiedqa_unifew',
                             #'unifiedqa_unifew_nli_mix',
                             #'unifiedqa_what_v2', 
                             #'unifiedqa_snli_mix_what_v2',
                             #'unifiedqa_snli_mix_what_with_choices_v2',
                             #'unifiedqa_ynm', 
                             #'unifiedqa_snli_mix_ynm', 
                             #'unifiedqa_snli_mix_ynm_with_choices'
                             ], 
                   'cos_e': [
                             #'squad',
                             #'record',
                             #'t5_fewshot_infilling_more_natural', 
                             #'t5_fewshot_infilling_with_choices',
                             'unifiedqa_matching'
                             ],
                    'ecqa': [
                             #'squad',
                             #'record',
                             #'t5_fewshot_infilling_more_natural', 
                             #'t5_fewshot_infilling_with_choices',
                             'unifiedqa_matching'
                             ],
                   'sensemaking': [
                                   #'t5_fewshot_infilling',
                                   #'t5_fewshot_infilling_bool',  
                                   #'copa_bool', 
                                   #'copa_with_question',
                                   #'record',
                                   #'squad_yn',
                                   #'squad_what',
                                   #'unifiedqa_yn', 'unifiedqa_yn_with_choices', 
                                   'unifiedqa_what',
                                   #'unifiedqa_what_with_choices',
                                   #'unifiedqa_what_no_tags',
                                   #'unifiedqa_yn_no_tags',
                                   #'squad_yn_no_tags',
                                   #'squad_what_no_tags'
                                   ], 
                   'sbic': [#'t5_fewshot_infilling', 
                            #'t5_fewshot_infilling_bool',
                            #'t5_fewshot_infilling_more_natural',
                            #'cola',
                            #'squad_yn',
                            #'squad_what',
                            #'unifiedqa_unifew', 
                            #'unified_qa_yn',
                            #'unified_qa_yn_with_choices', 
                            #'unified_qa_what',
                            #'unified_qa_what_with_choices',
                            #'squad_yn_with_tags',
                            #'squad_what_with_tags',
                            #'unified_qa_yn_with_tags',
                            #'unified_qa_yn_with_choices_and_tags',
                            'unified_qa_what_with_tags', 
                            #'unified_qa_what_with_choices_and_tags'
                            ]}

    n_shots_dict = {'esnli': [16],
                    'cos_e': [48],
                    'ecqa': [48],
                    'sensemaking': [24],
                    'sbic': [24]}

    data_path_dict = {'esnli': None,
                      'cos_e': None,
                      'ecqa': '../data/ECQA-Dataset',
                      'sensemaking': '../data/SenseMaking/',
                      'sbic': '../data/SBIC/'}

    commands = []
    for hparams in itertools.product(dataset_vals,
                                    seed_vals,
                                    model_vals,
                                    early_stopping_patience_vals,
                                    max_steps_vals,
                                    epochs_vals,
                                    warmup_steps_vals,
                                    learning_rate_vals,
                                    per_device_train_batch_size_vals,
                                    eval_steps_vals,
                                    fewshot_eval_size_vals,
                                    explanation_sep_vals,
                                    tokenizer_vals):
        dataset, seed, model, early_stopping_patience, max_steps, epochs, warmup_steps, learning_rate, per_device_train_batch_size, eval_steps, fewshot_eval_size, explanation_sep, tokenizer_name = hparams

        format_vals = format_dict[dataset]
        n_shot_vals = n_shots_dict[dataset]
        print("-----------------------")
        print(f"dataset = {dataset}")
        print("-----------------------")
        print(format_vals)
        print(n_shot_vals)
        print(hparams)
        run_name_parts = []

        # Iterate through hparams, excluding dictionary elements

        for format_n_shots_params in itertools.product(format_vals, n_shot_vals):
            format, n_shots = format_n_shots_params
            print(hparams)
            # if 't5' in model: #only perform unified format check for t5 models
            #     print(f"t5 in model")
            #     if ('unified' in model and 'unified' not in format) or ('unified' not in model and 'unified' in format):
            #         continue
            for v in hparams:
                if not isinstance(v, dict):
                    run_name_parts.append(str(v))  # Convert to string and store it
            print(run_name_parts)
            run_name = '-'.join([str(v) for v in hparams if not isinstance(v, dict)]).replace('/', '-').replace('.', '').replace(' ','')


            run_name += '-'.join([format, str(n_shots)])

            if not os.path.isdir(os.path.join(args.exp_root, run_name)):
                os.mkdir(os.path.join(args.exp_root, run_name))

            output_dir = os.path.join(args.exp_root, run_name)

            # You might need to adjust the cmd_prefix for the type of server you're using, e.g., if you use Slurm
            # We run the commands on google cloud with N gpus allocated just for this project 
            if args.deepspeed:
                # cmd_prefix = "PYTHONPATH=. deepspeed input_to_label_and_rationale_2.py "
                cmd_prefix = "deepspeed scripts/input_to_label_and_rationale_2.py "
                cmd_batch_size = f" --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps {per_device_train_batch_size} "
            else:
                # cmd_prefix = "PYTHONPATH=. python input_to_label_and_rationale_2.py "
                # cmd_prefix = "CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --debug scripts/input_to_label_and_rationale_2.py "
                # cmd_batch_size = f" --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 "

                cmd_prefix = "CUDA_VISIBLE_DEVICES=0 accelerate launch --debug scripts/input_to_label_and_rationale_2.py "
                cmd_batch_size = f" --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 "

            cmd = f'''{cmd_prefix} \
                    --output_dir {output_dir}  --model_type {model} --model_class {model}   \
                    --tokenizer_name {tokenizer_name}   --task_name {dataset}  --version v1.0 --do_train --dev_predict   \
                    --logging_first_step  --logging_steps 1  --save_total_limit 1  --seed {seed}     --num_train_epochs {epochs}    \
                    {cmd_batch_size} \
                    --early_stopping_patience {early_stopping_patience}   \
                    --n_shots {n_shots}  --fewshot_eval_size {fewshot_eval_size}   \
                    --learning_rate {learning_rate}  --warmup_steps {warmup_steps}  \
                    --io_format {format}  --explanation_sep {explanation_sep}  \
                    --max_steps {max_steps}  --lr_scheduler_type constant  --eval_steps {eval_steps}'''

            print("---------------------------------------")
            print(cmd)
            print("---------------------------------------")
            if args.deepspeed:
                cmd += " --deepspeed deepspeed_config.json"

            data_path = data_path_dict[dataset]
            if data_path:
                cmd += f" --data_path {data_path}"
            if args.use_gpt3:
                cmd += f" --use_gpt3  --gpt3_max_eval_size {args.gpt3_max_eval_size}"
                cmd = f'OPENAI_KEY={args.openai_key} {cmd}'
            commands.append(cmd)
    
    gpu_list = list(range(args.n_gpus))
    if args.not_dryrun:
        for cmd in commands:
            foo(cmd, gpu_list)
    else: 
        for cmd in commands:
            print (cmd)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", type=str, help="path to a directory where checkpoints will be saved")
    parser.add_argument("--not_dryrun", default=False, action='store_true', help="Actually run the experiments")
    parser.add_argument("--collect_results", default=False, action='store_true', help="Collect results from exp_root")
    parser.add_argument("--experiment_id", type=str, default="t5_unifiedqa_fewshot", help="Identifier that sets hyperparameters")
    parser.add_argument("--deepspeed", default=False, action='store_true', help="Use deepspeed")
    parser.add_argument("--n_gpus", type=int, default=4, help="The number of available GPUs")
    parser.add_argument("--model_vals", type=str, default=None, help="A string of models to evaluate separeted by comma, e.g., 't5-base,allenai/unifiedqa-t5-large'"
                                                                     "We used the following models for our experiments:"
                                                                     "t5-base"
                                                                     "t5-large"
                                                                     "t5-3b"
                                                                     "allenai/unifiedqa-t5-base"
                                                                     "allenai/unifiedqa-t5-large"
                                                                     "allenai/unifiedqa-t5-3b"
                                                                     "meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset_vals", type=str, default=None, help="A string of datasets for train/eval separeted by comma, e.g., 'ecqa,sensemaking'"
                                                                       "We used the following datasets for our experiments:"
                                                                       "ecqa"
                                                                       "sbic"
                                                                       "esnli"
                                                                       "sensemaking"
                                                                       "cos_e (don't recommend using it)") 
    
    parser.add_argument("--model_class", type=str, default='t5', help="What class model to be used. We used the following:"
                                                                    "t5"
                                                                    "llama")
    parser.add_argument("--use_gpt3", default=False, action='store_true', help="Use gpt3")
    parser.add_argument("--gpt3_max_eval_size", default=18, help="Number of evaluation samples per episode for gpt3")    
    parser.add_argument("--openai_key", type=str, help="Openai key")                                                     
    args = parser.parse_args()

    
    if args.collect_results:
        collect_results(args)
    else:
        if args.experiment_id != 't5_unifiedqa_fewshot':
            raise ValueError('We support only `t5_unifiedqa_fewshot` as `experiment_id`.')
        
        
        experiments[args.experiment_id]['model_vals'] = args.model_vals.replace(' ','').split(',')
        experiments[args.experiment_id]['tokenizer_vals'] = [model.replace('allenai/unifiedqa-','') for model in experiments['t5_unifiedqa_fewshot']['model_vals']]
        experiments[args.experiment_id]['dataset_vals'] = args.dataset_vals.replace(' ','').split(',')
        start_time = time.time()
        run_exp(args)
        total_time = time.time() - start_time
        total_time_minutes = total_time / 60.0
        total_time_hours = total_time_minutes / 60.0
        print (f"Train/eval of one model x one dataset x 60 train-test combinations x one value of # shots took x a few prompts took:\n\t {total_time_minutes} minutes / {total_time_hours} hours")
