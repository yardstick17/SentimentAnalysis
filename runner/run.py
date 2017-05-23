import logging

from prediction import Prediction
from train import TrainModel

SMALL_DATASET = 'small_dataset'
LARGE_DATASET = 'large_dataset'


def train(dataset):
    train_model = TrainModel(dataset)
    train_model.populate_text_model()


def prediction(dataset):
    prediction_model = Prediction(dataset)
    prediction_model.predict()
    prediction_model.write_result_to_file()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    for dataset in [SMALL_DATASET, LARGE_DATASET]:
        train(dataset=dataset)
        prediction(dataset=dataset)
