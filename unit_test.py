import pytest
from pathlib import Path
import main
from main import *
import yaml

@pytest.fixture
def example_path():
    return Path('/home/ddelahoz/Desarrollo/HDD/Opennmt/onmt-evaluation/directorio_prueba')

@pytest.fixture
def example_evaluation():
    evaluation = yaml.safe_load('''
    evaluation_name: newstest-2013
    datasets: [a, b, c]
    language-pair: en-es
    tokenizer_config:
    #This section must have a complete alingment with tokenizer parameters
        -   mode: conservative
            bpe_model_path:
            joiner_annotate: 
    models: [1, 2, 3]
    metrics: [1, 2, 3]
    save_directory: Prueba
    ''')
    return evaluation

@pytest.fixture
def example_dataset():
    dataset = yaml.safe_load('''
    name: newstest-2013
    original_direction: None
    files:
    -   ca: fichero_en_catalan.ca
        en: fichero_en_ingles.en
        es: fichero_en_español.es
    ''')
    return dataset

@pytest.fixture
def example_tokenizer():
    tokenizer_config = yaml.safe_load('''
    tokenizer_config:
        -   mode: conservative
            bpe_model_path: /home/ddelahoz/Desarrollo/HDD/Data/enes-32k.model
            joiner_annotate: true
    ''')
    return tokenizer_config
    
@pytest.fixture
def example_load_config():
    permanent_config = yaml.safe_load('''
    datasets:
    -   name: newstest-2013
        original_direction: None
        files:
        -   ca: fichero_en_catalan.ca
            en: fichero_en_ingles.en
            es: fichero_en_español.es
    -   name: anniebonnie
        original_direction: en-es
        files:
        -   en:
            es:
    ''')
    experiments_config = yaml.safe_load('''
    permanent_config: /home/ddelahoz/Desarrollo/HDD/Opennmt/onmt-evaluation/permanent_config.yaml
    translation_config:
        gpu: 0
        batch_size: 16384 
        batch_type: tokens 
        beam_size: 5 
        max_length: 300 
        replace_unk: True

    evaluations:
        evaluation_1:
            evaluation_name: newstest-2013
            datasets: [a, b, c]
            language-pair: en-es
            tokenizer_config:
            #This section must have a complete alingment with tokenizer parameters
                -   mode: conservative
                    bpe_model_path:
                    joiner_annotate: 
            models: [1, 2, 3]
            metrics: [1, 2, 3]
            save_directory: Prueba
        evaluation_2:
            evaluation_name:
            datasets: [a, b, c]
            language-pair: 
            tokenizer:
            models: []
            metrics: []
            save_results:    
    ''')

    return({'evaluation_config' : experiments_config,
            'permanent_config' : permanent_config})

@pytest.fixture
def dataset_example():
    return yaml.safe_load('''
    name: newstest-2013
      original_direction: None
      available_languages: [ca, en, es]
      files:
        - ca: fichero_en_catalan.ca
          en: fichero_en_ingles.en
          es: "fichero_en_español.es"
    ''')

@pytest.fixture
def mock_config():
    class Object(object):
        pass
    config = Object()
    config.config = Path('/home/ddelahoz/Desarrollo/HDD/Opennmt/onmt-evaluation/Prueba.yaml')
    return config


def test_example_load_config(example_load_config, mock_config):
    config = load_config(mock_config)
    assert config == example_load_config


def test_create_directory(example_path):
    create_directory(example_path)
    assert example_path.is_dir() == True
    create_directory(example_path)
    os.rmdir(example_path)

def test_get_languages():
    one_language = 'es'
    two_language = 'en-es'
    three_language = 'en-es-ca'
    assert get_languages(one_language) == ['es']
    assert get_languages(two_language) == ['en', 'es']
    assert get_languages(three_language) == ['en', 'es', 'ca']

def test_get_available_datasets(example_load_config):
    datasets = example_load_config['permanent_config']['datasets']
    result = get_available_datasets(datasets)
    assert result == ['newstest-2013', 'anniebonnie']

def test_get_test_files(example_load_config):
    dataset_config = example_load_config['permanent_config']['datasets'][0]
    valid_languages = ['es', 'ca']
    one_language = ['es']
    invalid_languages = ['en, st, ca']
    invalid_one_language = ['st']
    assert get_test_files(dataset_config, valid_languages) == ['fichero_en_español.es', 'fichero_en_catalan.ca']
    assert get_test_files(dataset_config, one_language) == ['fichero_en_español.es']
    
def test_valid_language_pair_for_dataset(example_load_config):
    dataset_config = example_load_config['permanent_config']['datasets'][0]
    valid_languages = 'es-ca'
    one_language = 'es'
    invalid_languages = 'st-ca'
    invalid_one_language = 'st'
    assert valid_language_pair_for_dataset(dataset_config, valid_languages) != None
    assert valid_language_pair_for_dataset(dataset_config, one_language) != None
    assert valid_language_pair_for_dataset(dataset_config, invalid_languages) == None
    assert valid_language_pair_for_dataset(dataset_config, invalid_one_language) == None

def test_get_dataset(example_load_config, example_dataset):
    dataset_name = example_load_config['permanent_config']['datasets'][0]['name']
    permanent_config = example_load_config['permanent_config']
    assert get_dataset(dataset_name, permanent_config) == example_dataset
    assert get_dataset('Patata', permanent_config) == None

def test_generate_output_directory(example_evaluation):
    output_path = generate_output_directory(example_evaluation)
    assert output_path == Path(Path('Prueba')/'newstest-2013')

def test_is_evaluation():
    evaluation = ['es', 'ca']
    inference = ['es']
    assert is_evaluation(evaluation) is True
    assert is_evaluation(inference) is False

def test_write_metric_in_log():
    metric = 'BLEU'
    metric_value = 32.23
    save_directory = Path('/home/ddelahoz/Desarrollo/HDD/Opennmt/onmt-evaluation')
    if os.path.isfile(Path(save_directory / 'result.out')):
        os.remove(Path(save_directory / 'result.out'))
    write_metric_in_log(metric, metric_value, save_directory)
    with open(Path(save_directory / 'result.out'), 'r', encoding='utf-8') as f:
        result = f.read()
    assert result == f'Result of BLEU metric\n32.23\n'
    os.remove(Path(save_directory / 'result.out'))

def test_get_dataset_config(example_load_config):
    datasets = example_load_config['permanent_config']['datasets']
    valid_dataset = 'newstest-2013'
    invalid_dataset = 'Clarita-2013'
    assert get_dataset_config(valid_dataset, datasets) == datasets[0]
    assert get_dataset_config(invalid_dataset, datasets) == None

def generate_tokenized_paths(files):
    return [str(file) + '.bpe' for file in files]

def test_tokenize_dataset(example_tokenizer):
    correct_files = [Path('/home/ddelahoz/Desarrollo/HDD/Opennmt/onmt-evaluation/newstest-2013/newstest-2013.ca'), \
        Path('/home/ddelahoz/Desarrollo/HDD/Opennmt/onmt-evaluation/newstest-2013/newstest-2013.es')]
    correct_file = [Path('/home/ddelahoz/Desarrollo/HDD/Opennmt/onmt-evaluation/newstest-2013/newstest-2013.ca')]

    tokenizer_config = example_tokenizer['tokenizer_config'][0]
    target_correct_files = generate_tokenized_paths(correct_files)
    target_correct_file = generate_tokenized_paths(correct_file)

    tokenize_dataset(correct_files, tokenizer_config)
    assert all(os.path.exists(file) for file in target_correct_files) is True
    for file in target_correct_files:
        if os.path.exists(file):
            os.remove(file)

    tokenize_dataset(correct_file, tokenizer_config)
    assert os.path.exists(target_correct_file[0]) is True
    if os.path.exists(target_correct_file[0]):
        os.remove(target_correct_file[0])

def test_detokenized_dataset(example_tokenizer):
    tokenizer_config = example_tokenizer['tokenizer_config'][0]
    correct_file = generate_tokenized_paths([Path('/home/ddelahoz/Desarrollo/HDD/Opennmt/onmt-evaluation/newstest-2013/newstest-2013.en')])[0]
    detokenize_result(correct_file, tokenizer_config)
    assert os.path.exists(correct_file[:-4]+'.out')
    os.remove(correct_file[:-4]+'.out')






