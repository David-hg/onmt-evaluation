import pytest
from pathlib import Path
import main
from main import *
import yaml

@pytest.fixture
def example_path():
    return Path('E:\\Codigo\\Model_evaluation\\directorio_prueba')

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
def example_config():
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
    permanent_config: E:\\Codigo\\Model_evaluation\\permanent_config.yaml
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
    config.config = Path('E:\\Codigo\\Model_evaluation\\Prueba.yaml')
    return config


def test_load_config(example_config, mock_config):
    print(example_config['permanent_config'])
    config = load_config(mock_config)
    print(config['permanent_config'])
    assert config == example_config


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

def get_dataset_config(example_config):
    dataset_name = example_config['evaluation_config']['evaluations']['evaluation_1']['evaluation_name']
    datasets = example_config['permanent_config']['datasets']
    result = get_dataset_config(dataset_name, datasets)
    assert result == datasets[0]

def test_get_available_datasets(example_config):
    datasets = example_config['permanent_config']['datasets']
    result = get_available_datasets(datasets)
    assert result == ['newstest-2013', 'anniebonnie']

def test_get_test_files(example_config):
    dataset_config = example_config['permanent_config']['datasets'][0]
    valid_languages = ['es', 'ca']
    one_language = ['es']
    invalid_languages = ['en, st, ca']
    invalid_one_language = ['st']
    assert get_test_files(dataset_config, valid_languages) == ['fichero_en_español.es', 'fichero_en_catalan.ca']
    assert get_test_files(dataset_config, one_language) == ['fichero_en_español.es']
    
def test_valid_language_pair_for_dataset(example_config):
    dataset_config = example_config['permanent_config']['datasets'][0]
    valid_languages = 'es-ca'
    one_language = 'es'
    invalid_languages = 'st-ca'
    invalid_one_language = 'st'
    assert valid_language_pair_for_dataset(dataset_config, valid_languages) != None
    assert valid_language_pair_for_dataset(dataset_config, one_language) != None
    assert valid_language_pair_for_dataset(dataset_config, invalid_languages) == None
    assert valid_language_pair_for_dataset(dataset_config, invalid_one_language) == None

def test_get_dataset(example_config, example_dataset):
    dataset_name = example_config['permanent_config']['datasets'][0]['name']
    permanent_config = example_config['permanent_config']
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
    save_directory = Path('E:\\Codigo\\Model_evaluation')
    if os.path.isfile(Path(save_directory / 'result.out')):
        os.remove(Path(save_directory / 'result.out'))
    write_metric_in_log(metric, metric_value, save_directory)
    with open(Path(save_directory / 'result.out'), 'r', encoding='utf-8') as f:
        result = f.read()
    assert result == f'Result of BLEU metric\n32.23\n'
    os.remove(Path(save_directory / 'result.out'))




