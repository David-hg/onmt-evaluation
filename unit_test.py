import pytest
from pathlib import Path
import main
from main import *
import yaml

@pytest.fixture
def example_path():
    return Path('directorio_prueba')

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
            bpe_model_path: test/enes-32k.model
            joiner_annotate: true
    ''')
    return tokenizer_config

@pytest.fixture
def example_translation_config():
    return yaml.safe_load('''
        gpu: 0
        batch_size: 16384 
        batch_type: tokens 
        beam_size: 5 
        max_length: 300 
        replace_unk: True
    ''')
    
@pytest.fixture
def example_load_config():
    permanent_config = yaml.safe_load('''
    datasets:
    - name: newstest-2013
      original_direction: None
      files:
        - ca: test/newstest-2013/newstest-2013.ca
          en: test/newstest-2013/newstest-2013.en
          es: test/newstest-2013/newstest-2013.es
    - name: anniebonnie
      original_direction: en-es
      files:
        - en:
          es:
    ''')
    experiments_config = yaml.safe_load('''
    permanent_config: permanent_config.yaml
    translation_config:
        gpu: 0
        batch_size: 16384 
        batch_type: tokens 
        beam_size: 5 
        max_length: 300 
        replace_unk: True

    evaluations:
        evaluation_1:
            evaluation_name: eval-1
            datasets: [newstest-2013, newstest-2013]
            language-pair: en-es
            tokenizer_config:
            #This section must have a complete alingment with tokenizer parameters
                -   mode: conservative
                    bpe_model_path: test/enes-32k.model
                    joiner_annotate: true
            models: [test/testing_model_en_es.pt]
            metrics: [BLEU, CHRF, COMET]
            save_directory: Prueba
        evaluation_2:
            evaluation_name: eval-2
            datasets: [newstest-2013, newstest-2013]
            language-pair: en-es
            tokenizer_config:
            #This section must have a complete alingment with tokenizer parameters
                -   mode: conservative
                    bpe_model_path: test/enes-32k.model
                    joiner_annotate: true
            models: [test/testing_model_en_es.pt]
            metrics: [COMET]
            save_directory: Prueba    
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
    config.config = Path('Prueba.yaml')
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
    save_directory = Path('')
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
    correct_files = [Path('test/newstest-2013/newstest-2013.ca'), \
        Path('test/newstest-2013/newstest-2013.es'), Path('test/newstest-2013/newstest-2013.en')]
    correct_file = [Path('test/newstest-2013/newstest-2013.ca')]

    tokenizer_config = example_tokenizer['tokenizer_config'][0]
    target_correct_files = generate_tokenized_paths(correct_files)
    target_correct_file = generate_tokenized_paths(correct_file)

    tokenize_dataset(correct_files, tokenizer_config)
    assert all(os.path.exists(file) for file in target_correct_files) is True
    for file in target_correct_files:
        if os.path.exists(file) and file != 'test/newstest-2013/newstest-2013.en.bpe':
            #os.remove(file)
            pass

    tokenize_dataset(correct_file, tokenizer_config)
    assert os.path.exists(target_correct_file[0]) is True
    if os.path.exists(target_correct_file[0]):
        #os.remove(target_correct_file[0])
        pass

def test_detokenized_dataset(example_tokenizer):
    tokenizer_config = example_tokenizer['tokenizer_config'][0]
    correct_file = generate_tokenized_paths([Path('test/newstest-2013/newstest-2013.en')])[0]
    detokenize_result(correct_file, tokenizer_config)
    print(correct_file[:-4])
    assert os.path.exists(correct_file[:-4])
    #os.remove(correct_file[:-4])

def test_compute_bleu_or_chrf():
    bleu = 'BLEU'
    chrf = 'CHRF'
    hypothesis_file = Path('test/newstest-2013/newstest-2013.en.out')
    target_file = [Path('test/newstest-2013/newstest-2013.es')]
    assert 'BLEU' in str(compute_bleu_or_chrf(hypothesis_file, target_file, bleu))
    assert 'chrF2++' in str(compute_bleu_or_chrf(hypothesis_file, target_file, chrf))

def test_compute_metrics():
    testing_files = [Path('test/newstest-2013/newstest-2013.ca'), Path('test/newstest-2013/newstest-2013.es')]
    hypothesis_file = Path('test/newstest-2013/newstest-2013.es')
    metrics = ['COMET']
    save_directory = Path('test/newstest-2013')
    compute_metrics(hypothesis_file, testing_files, metrics, save_directory)
    assert os.path.isfile(Path(save_directory / 'result.out'))
    assert None
    if os.path.isfile(Path(save_directory / 'result.out')):
        os.remove(Path(save_directory / 'result.out'))

def test_translate_dataset(example_translation_config):
    model_config = example_translation_config
    tokenized_files = [Path('test/newstest-2013/newstest-2013.en.bpe')]
    models = ['test/testing_model_en_es.pt']
    saving_path = Path('test/newstest-2013')
    translate_dataset(model_config, tokenized_files, models, saving_path)
    assert os.path.isfile('test/newstest-2013/newstest-2013.en.out.bpe')
    if os.path.isfile(Path('test/newstest-2013/newstest-2013.en.out.bpe')):
        os.remove(Path('test/newstest-2013/newstest-2013.en.out.bpe'))

def test_main(mock_config):
    main(mock_config)

def test_cleaning():
    for file in os.listdir(Path('test/newstest-2013')):
        if file[-4:] == '.bpe':
            os.remove('test/newstest-2013/' + file)





