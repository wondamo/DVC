import argparse
import yaml
import json
from pathlib import Path
import pandas as pd
from typing import Text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model



def evaluate(config_path: Text) -> None:
    """Preprocess the text data.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    train = pd.read_csv(config['data']['train_path'])
    test = pd.read_csv(config['data']['test_path'])

    X_train = train['processed_X']
    X_test, y_test = test['processed_X'], test['processed_y']

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    X_test = tokenizer.texts_to_sequences(X_test)

    maxlen=200
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    
    model = load_model(config['train']['model_checkpoint'])

    accuracy = model.evaluate(X_test, y_test)[-1]
    
    json.dump(
        obj={"accuracy": accuracy},
        fp=open(config['evaluate']['metrics_file'], 'w')
    )


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate(args.config)