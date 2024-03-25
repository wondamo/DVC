import argparse
import yaml
import spacy
import pandas as pd
import numpy as np
from typing import Text
from spacy.lang.en.stop_words import STOP_WORDS
from src.utils.logs import get_logger

def process(config_path: Text) -> None:
    """Preprocess the text data.
    Args:
        config_path {Text}: path to config
    """
    nlp = spacy.load("en_core_web_sm")

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('PREPROCESS', log_level=config['base']['log_level'])

    logger.info("Load raw data")
    data = pd.read_csv(config['data']['path'])
    data = data.dropna(axis=0, subset=['reviewText'])

    def preprocess(string):
        doc = nlp(string)
        lemma = [token.lemma_ for token in doc if token.lemma_.isalpha() or token.lemma_ not in STOP_WORDS]
        return ' '.join(lemma)
    
    logger.info("Preprocess text data")
    data['processed_X'] = [preprocess(sen) for sen in list(data['reviewText'])]
    data['processed_y'] = np.array(data['overall'].map({1:0, 2:0, 3:1, 4:1, 5:1}))

    logger.info("Save preprocessed data")
    data.to_csv(config['data']['processed_path'], index=False)


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    process(args.config)