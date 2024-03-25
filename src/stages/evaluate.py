import argparse
import yaml
import json
from pathlib import Path
import pandas as pd
from typing import Text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from src.utils.logs import get_logger



def evaluate(config_path: Text) -> None:
    """Preprocess the text data.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('EVALUATE', log_level=config['base']['log_level'])

    logger.info("Load train dataset")
    train = pd.read_csv(config['data']['train_path'])
    logger.info("Load test dataset")
    test = pd.read_csv(config['data']['test_path'])

    X_train = train['processed_X']
    X_test, y_test = test['processed_X'], test['processed_y']

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    X_test = tokenizer.texts_to_sequences(X_test)

    maxlen=200
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    
    logger.info("Load model")
    model = load_model(config['train']['model_checkpoint'])

    logger.info("Evaluate model")
    accuracy = model.evaluate(X_test, y_test)[-1]
    
    logger.info("Save metrics")
    json.dump(
        obj={"accuracy": accuracy},
        fp=open(config['evaluate']['metrics_file'], 'w')
    )
    logger.info("Accuracy metrics file saved to : {config['evaluate']['metrics_file']}")


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate(args.config)