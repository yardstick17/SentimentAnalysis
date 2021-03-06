import logging
from itertools import count, groupby

import h5py
import joblib
import numpy as np
import yaml
from gensim import matutils
from pandas import DataFrame
from sklearn.externals import joblib

from dataset.pre_process_dataset import read_content
from py_hdf5.hdf5 import append_batch_to_h5_dataset
from text_model.bow_text_model import TextModel


class Prediction:
    PREDECTION_DATASET = 'prediction_result'

    def __init__(self, dataset):
        self.dataset = dataset
        self.config = self.read_dataset_config(dataset)
        self.prediction_model = self.load_prediction_model()
        self.text_model = self.load_text_model()

    def load_prediction_model(self):
        logging.info('Loading prediction model')
        model_filepath = self.config['model_file']
        return joblib.load(model_filepath)

    @staticmethod
    def read_dataset_config(dataset):
        with open("config.yml", 'r') as ymlfile:
            config = yaml.load(ymlfile)
        return config[dataset]

    def load_text_model(self):
        logging.info('Loading tfidf model')
        tfidf_filepath = self.config['tf_idf']
        return TextModel.load(tfidf_filepath)

    def predict(self):
        prediction_file = self.config['prediction']['dataset_filepath']
        test_filestream = read_content(prediction_file)
        hf = self.get_hdf5('w')
        for index, chunk_data in enumerate(self.split_every(10000, test_filestream)):

            logging.info('Predicting for chunk: {}'.format(index))

            sentence_list = [text[0] for text in chunk_data]

            vector = self.text_model.texts_to_vector(sentence_list)

            vector = np.asarray([matutils.sparse2full(vec, self.text_model.dictionary_length) for vec in vector])
            chunk_predictions = self.prediction_model.predict(vector)
            append_batch_to_h5_dataset(hf, self.PREDECTION_DATASET, chunk_predictions, (None,))

    def get_hdf5(self, mode='r'):
        return h5py.File(self._h5py_filename, mode=mode)

    def write_result_to_file(self):
        hf = self.get_hdf5()
        result_dataset = hf[self.PREDECTION_DATASET]
        result = (r for r in result_dataset)
        df = DataFrame(result)
        file = self.config['prediction']['prediction_result_filepath'] + '.csv'
        df.to_csv(file, chunksize=1000, mode='a')

    @staticmethod
    def split_every(size, iterable):
        c = count()
        for k, g in groupby(iterable, lambda x: next(c) // size):
            yield list(g)  # or yield g if you want to output a generator

    @property
    def _h5py_filename(self):
        return self.config['prediction']['prediction_result_filepath'] + '.h'
