import argparse
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Text


def split(config_path: Text) -> None:
    """Split dataset into train/test.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    data = pd.read_csv(config['data']['processed_path'])
    data.dropna(axis=0, subset=['reviewText'])

    train, test = train_test_split(data, test_size=0.1, random_state=config['base']['random_state'])
    train.to_csv(config['data']['train_path'], index=False)
    test.to_csv(config['data']['test_path'], index=False)


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    split(args.config)