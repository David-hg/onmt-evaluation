import argparse
import os
import subprocess

import pyonmttok
import yaml
from yaml.loader import SafeLoader
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF

parser = argparse.ArgumentParser()

parser.add_argument("-c", "--config", help = "Config file with all parameters")

args = parser.parse_args()

def load_config(args):

    with open(args.config, encoding = 'utf-8') as f:
        config = yaml.load(f, Loader=SafeLoader)
    
    with open(config['permanent_config'], encoding = 'utf-8') as f:
        permanent_config = yaml.load(f, Loader=SafeLoader)

    return({'evaluation_config' : config,
            'permanent_config' : permanent_config})

def create_directory(path):
    normalized_path = Path(path)
    if not Path(normalized_path).is_dir():
        print(f'Path: {normalized_path} does not exist')
        os.makedirs(Path(normalized_path))
        print(f'Created path: {normalized_path}')

def get_test_files(dataset_config, languages):
    return [dataset_config['files'][0][language] for language in languages]

def get_languages(language_pair):
    # Languages are splited based on the - character. In case of a dataset without reference
    # the language indicated in the evaluations.yaml will be a returned in a one item list
    return language_pair.split('-')

def get_dataset_config(dataset_name, datasets):
    for dataset in datasets:
        if dataset_name == dataset['name']:
            return dataset
    print(f'Dataset {dataset_name} is not available, available datasets are: {get_available_datasets(datasets)}')
    return None

def get_available_datasets(datasets):
    return [dataset['name'] for dataset in datasets]

def valid_language_pair_for_dataset(dataset_config, language_pair):
    languages = get_languages(language_pair)
    dataset_languages = [language for language in dataset_config['files'][0]]
    if all(language in dataset_languages for language in languages):
        return [dataset_config['files'][0][language] for language in languages]
    else:
        #print(f'''Some of the languages you gave {languages} are not available for dataset {dataset_config['name']}
                #this are the available ones: {dataset_languages}''')
        return None


def tokenize_dataset(testing_files, tokenizer_config):
    #TODO: tokenizer function to tokenize a file given a tokenizer
    files = []
    tokenizer = pyonmttok.Tokenizer(**tokenizer_config)
    for file in testing_files:
        tokenizer.tokenize_file(f"{file}", f"{file}.bpe")
        files.append(Path(f"{file}.bpe"))
    return files

def detokenize_result(inference_file_tokenized, tokenizer_config):
    #TODO: detokenizer function to detokenize a file given a tokenizer
    print(inference_file_tokenized)
    inference_file_tokenized = str(inference_file_tokenized)
    tokenizer = pyonmttok.Tokenizer(**tokenizer_config)
    detokenized_file = tokenizer.detokenize_file(f"{inference_file_tokenized}", f"{inference_file_tokenized[:-4]}")
    return f"{inference_file_tokenized[:-4]}"
    

def get_dataset(dataset_name, permanent_config):
    for dataset in permanent_config['datasets']:
        if dataset['name'] is dataset_name:
            return dataset
    print(f'Dataset {dataset_name} is not available')
    return None

def generate_output_directory(evaluation_config):
    output_path = Path(evaluation_config['save_directory']) / evaluation_config['evaluation_name']
    return output_path

def translate_dataset(model_config, tokenized_files, models, saving_path):
    output_file_name = os.path.splitext(tokenized_files[0].parts[-1])[0] + '.out.bpe'
    output_file = saving_path/output_file_name
    config_string = 'onmt_translate '
    for parameter in model_config:
        if parameter == 'replace_unk':
            config_string += '--' + parameter + ' '
        else:
            config_string += '--' + parameter + ' ' + str(model_config[parameter]) + ' '
    config_string += '--model '
    for model in models:
        config_string += model + ' '
    for file, atribute in zip(tokenized_files, ['-src', '-tgt']):
        config_string += f'{atribute} {file} '
    config_string +=  f'--output {output_file}'
    os.system(config_string)
    return output_file

def is_evaluation(languages):
    return len(languages)==2

def translate(model_config, evaluation_config, dataset_config):
    languages = get_languages(evaluation_config['language-pair'])
    testing_files = get_test_files(dataset_config, languages)
    tokenized_files = tokenize_dataset(testing_files, evaluation_config['tokenizer_config'][0])
    output_directory = generate_output_directory(evaluation_config)
    create_directory(output_directory)
    translated_tokenized = translate_dataset(model_config, tokenized_files, evaluation_config['models'],output_directory)
    hypothesis_file = detokenize_result(translated_tokenized, evaluation_config['tokenizer_config'][0])
    if is_evaluation(languages):
        evaluation_path = compute_metrics(hypothesis_file, testing_files, evaluation_config['metrics'], output_directory)
        return evaluation_path 
    return hypothesis_file

def write_metric_in_log(metric, metric_result, save_directory):
    with open(Path(save_directory / 'result.out'), 'a') as result_file:
        result_file.write(f'Result of {metric} metric\n')
        result_file.write(str(metric_result) + '\n')
    return Path(save_directory / 'result.out')
        
def compute_bleu_or_chrf(hypothesis_file, target_files, metric):
    metric = BLEU() if metric == 'BLEU' else CHRF(word_order=2)
    references = []
    with open(hypothesis_file, 'r', encoding = 'utf-8') as file:
        hypothesis = [sentence.strip() for sentence in file]
    
    for file_path in target_files:
        with open(file_path, 'r', encoding = 'utf-8') as file:
            references.append([sentence.strip() for sentence in file])
            
    result = metric.corpus_score(hypothesis, references)
    return result

def compute_metrics(hypothesis_file, testing_files, metrics, save_directory):
    hypothesis_file = hypothesis_file
    source_file = testing_files[0]
    target_files = testing_files[1:]
    log_file = None
    for metric in metrics:
        if metric == 'COMET':
            command = f'comet-score -s {source_file} -t {hypothesis_file} -r {target_files[0]} --quiet'.split()
            metric_value = subprocess.run(command, capture_output = True).stdout.decode('utf-8')
            log_file = write_metric_in_log(metric, metric_value, save_directory)
        elif metric in ['BLEU', 'CHRF']:
            metric_value = compute_bleu_or_chrf(hypothesis_file, target_files, metric)
            log_file = write_metric_in_log(metric, metric_value, save_directory)
        else:
            print(f'Sorry, metric: {metric} is not available currently suported metrics are BLEU CHRF and COMET')
    return log_file

def execute_evaluation(model_config, evaluation_config, permanent_config):
    for dataset_name in evaluation_config['datasets']:
        dataset_config = get_dataset_config(dataset_name, permanent_config['datasets'])
        if valid_language_pair_for_dataset(dataset_config, evaluation_config['language-pair']):
            output_file = translate(model_config, evaluation_config, dataset_config)
            print(f'Results saved in {output_file}')


def evaluate(config):
    permanent_config = config['permanent_config']
    model_config = config['evaluation_config']['translation_config']
    for evaluation in config['evaluation_config']['evaluations']:
        evaluation_config = config['evaluation_config']['evaluations'][evaluation]
        execute_evaluation(model_config, evaluation_config, permanent_config)
            

def main(args):
    config = load_config(args)
    print(config)
    #evaluate(config)

#main(args)