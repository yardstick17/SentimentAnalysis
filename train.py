import logging
import os
from itertools import count, groupby

import numpy as np
import yaml
from gensim import matutils
from sklearn.externals import joblib
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report

from dataset.pre_process_dataset import read_content
from text_model.bow_text_model import TextModel
from sklearn.utils import shuffle


class TrainModel:
    def __init__(self, dataset):
        self.config = self.read_dataset_config(dataset)

    @staticmethod
    def read_dataset_config(dataset):
        with open("config.yaml", 'r') as ymlfile:
            config = yaml.load(ymlfile)
        return config[dataset]

    @staticmethod
    def even_distribution(X, Y):
        X_0 = []
        X_1 = []

        for x, y in zip(X, Y):
            if y == 0:
                X_0.append(x)
            else:
                X_1.append(x)

        logging.info('{} {} Len of 0 and 1 labeled data'.format(len(X_0), len(X_1)))
        m = min(len(X_0), len(X_1))

        for i in range(3):
            X_0 = np.random.permutation(X_0)
            x_1, x_0 = shuffle(X_1[:m], X_0[:m])
            logging.info('After shuffling : {} {} Len of 0 and 1 labeled data'.format(len(x_0), len(x_1)))
            x_0 = list(x_0)
            x_1 = list(x_1)
            y = [1] * len(x_1)
            y.extend([0] * len(x_0))
            x_1.extend(x_0)
            logging.info('Returning data for training  : {} {} Len of X and Y labeled data'.format(len(x_1), len(y)))
            yield np.array(x_1), np.array(y)

    @staticmethod
    def get_streaming_data(training_file):
        training_datastream = read_content(filename=training_file)
        return training_datastream

    @staticmethod
    def split_every(size, iterable):
        c = count()
        for k, g in groupby(iterable, lambda x: next(c) // size):
            yield list(g)  # or yield g if you want to output a generator

    @staticmethod
    def get_classifier():
        return PassiveAggressiveClassifier()

    def populate_text_model(self):
        tf_idf_file = self.config['tf_idf']
        training_file = self.config['training']['dataset_filepath']
        trained_model_file = self.config['model_file']

        training_datastream = self.get_streaming_data(training_file)
        if os.path.isfile(tf_idf_file):
            text_model = TextModel.load(tfidf_filename=tf_idf_file)
        else:
            text_model = TextModel(tfidf_filename=tf_idf_file)
            text_model.add_documents(training_datastream)

        logging.info('Model Loaded')

        num_features = text_model.dictionary_length
        corpus_tfidf_stream = text_model.get_corpus_tfidf_stream()

        corpus_label_stream = text_model._get_corpus_label_stream()
        logging.info('Streaming data is set!!')
        clf = self.get_classifier()
        try:
            for index, chunk_data in enumerate(zip(self.split_every(11000, corpus_tfidf_stream),
                                                   self.split_every(11000, corpus_label_stream))):
                chunk_tfidf, chunk_label = chunk_data
                logging.info('Training on chunk: {}'.format(index))
                chunk_text_vector = chunk_tfidf

                X = np.asarray([matutils.sparse2full(vec, num_features) for vec in chunk_text_vector])
                Y = np.asarray(list(chunk_label))

                for X, y in self.even_distribution(X, Y):
                    logging.debug('Training  : {} {} Len of X and Y labeled data'.format(len(X), len(y)))
                    clf.partial_fit(X, y, classes=[0, 1])
                    y_pred = clf.predict(X)
                    logging.info('\n%s' % classification_report(y, y_pred=y_pred))

        except KeyboardInterrupt as e:
            logging.info('Patience is everything. Exception {}'.format(e))
        finally:
            logging.info('Saving the classifier for persistence, {}'.format(trained_model_file))
            joblib.dump(clf, trained_model_file)
            clf = joblib.load(filename=trained_model_file)
            logging.info('Sample predicting from the saved model %s ..' % clf.predict(X)[:5])

        logging.info('Model trained :) ')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
