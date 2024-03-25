import argparse
import yaml
import pandas as pd
from typing import Text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.report.visualize import plot_model_accuracy, plot_model_loss
from src.train.train import train_model
from src.utils.logs import get_logger


def train(config_path: Text) -> None:
    """Train model.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('TRAIN', log_level=config['base']['log_level'])

    logger.info("Load train dataset")
    train = pd.read_csv(config['data']['train_path'])
    X_train, y_train = train['processed_X'], train['processed_y']

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)

    vocab_size = len(tokenizer.word_index)
    
    maxlen=200
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    
    logger.info("Train model")
    model, history = train_model(X_train, y_train, vocab_size, config)

    logger.info("Save model accuracy plot")
    plt = plot_model_accuracy(history)
    plt.savefig(config['train']['model_accuracy_path'])
    logger.info("Model accuracy plot saved to : {config['train']['model_accuracy_path']}")
    
    logger.info("Save model accuracy plot")
    plt = plot_model_loss(history)
    plt.savefig(config['train']['model_loss_path'])
    logger.info("Accuracy metrics file saved to : {config['train']['model_loss_path']}")


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train(args.config)