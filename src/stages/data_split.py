import argparse
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Text
from src.utils.logs import get_logger


def split(config_path: Text) -> None:
    """Split dataset into train/test.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('DATA_SPLIT', log_level=config['base']['log_level'])

    logger.info("Load processed data")
    data = pd.read_csv(config['data']['processed_path'])
    data.dropna(axis=0, subset=['reviewText'])

    logger.info("Split processed data into train and test sets")
    train, test = train_test_split(data, test_size=config['data']['test_size'], random_state=config['base']['random_state'])

    logger.info("Save train and test sets")
    train.to_csv(config['data']['train_path'], index=False)
    test.to_csv(config['data']['test_path'], index=False)


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    split(args.config)