# -*- coding: utf-8 -*-
import logging
import operator
import os
import tempfile

import h5py
import tqdm
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from nltk import ngrams

from text_model.iterator_executor import IteratorExecutor
from py_hdf5.hdf5 import append_batch_to_h5_dataset

_dir_name = None

max_dictionary_size = 50000

N_GRAM = 2
CHUNK_SIZE = 1000


class TextModel:
    TEXT_DATASET = 'doc'
    TEXT_ID_DATASET = 'doc_id'
    TEXT_CATEGORY = 'doc_category'

    def __init__(self, tfidf_filename):
        self.tfidf_filename = tfidf_filename
        self.dictionary = None
        self.tfidf = None

    @property
    def dictionary_length(self):
        assert self.dictionary is not None
        return len(self.dictionary)

    @classmethod
    def load(cls, tfidf_filename):
        obj = cls(tfidf_filename=tfidf_filename)
        obj.dictionary = Dictionary.load(obj._dictionary_filename)
        obj.tfidf = TfidfModel.load(obj._tfidf_model_filename)
        return obj

    def get_hdf5(self, mode='r'):
        return h5py.File(self._h5py_filename, mode=mode)

    def texts_to_vector(self, texts: list):
        assert self.dictionary is not None
        assert self.tfidf is not None
        corpus = self._raw_text_to_ngram_bow(texts)
        return self.tfidf[corpus]

    def index_id_to_doc_id(self, index_id):
        hf = self.get_hdf5()  # TODO: Cache?
        return hf[self.TEXT_ID_DATASET][index_id]

    def _raw_text_to_ngram_bow(self, texts):
        cleaned_texts = map(self._clean_str, texts)
        return self._cleaned_text_to_ngram_bow(cleaned_texts)

    def _cleaned_text_to_ngram_bow(self, cleaned_texts):
        ngram_lists = map(self._text_to_ngram, cleaned_texts)
        return map(self.dictionary.doc2bow, ngram_lists)

    def add_documents(self, document_data_stream):
        if not os.path.isfile(self._h5py_filename):
            hf = self.get_hdf5('w')
            index = 0
            with IteratorExecutor() as executor:
                results = executor.map(TextModel.clean_str, document_data_stream, submit_chunk_size=CHUNK_SIZE)

                for category, text in tqdm.tqdm(results, desc='Number of items read from process pool'):
                    append_batch_to_h5_dataset(hf, self.TEXT_ID_DATASET, [index], (None,))
                    append_batch_to_h5_dataset(hf, self.TEXT_DATASET, [text.encode()], (None,), dtype="S10000")
                    append_batch_to_h5_dataset(hf, self.TEXT_CATEGORY, [category.encode()], (None,))
                    index += 1

            corpus_text_stream = self._get_corpus_text_stream()
            corpus_ngram_stream = map(self._text_to_ngram, corpus_text_stream)

            logging.info('Building bow for given corpus')
            self.dictionary = Dictionary(corpus_ngram_stream,
                                         prune_at=1 * int(max_dictionary_size))  # 170000(NUM_OF_WORDS_IN_ENGLISH)^2
            self.dictionary.save(self._dictionary_filename)
            ngram_bow_stream = self._get_corpus_bow_stream()
            logging.info('Building TFIDF model from bow(n-gram)')
            self.tfidf = TfidfModel(ngram_bow_stream)
            self.tfidf.save(self._tfidf_model_filename)

    def _text_to_ngram(self, text, n_gram=N_GRAM):
        return [' '.join(tup) for tup in ngrams(text.lower().split(), n_gram)]

    def _get_corpus_text_stream(self):
        corpus_stream = self._get_corpus()
        corpus_text_stream = map(operator.itemgetter(1), corpus_stream)
        corpus_text_stream = map(self.decode_strings, corpus_text_stream)
        return corpus_text_stream

    @staticmethod
    def decode_strings(byte_text):
        try:
            text = byte_text.decode('utf-8')
        except UnicodeDecodeError:
            text = byte_text.decode('latin-1')  # If error, it'll raise exceptions here
        return text

    def _get_corpus(self):
        with h5py.File(self._h5py_filename, mode='r') as hf:
            cleaned_text_doc_dset = hf[self.TEXT_DATASET]
            cleaned_doc_id_dset = hf[self.TEXT_ID_DATASET]
            category_dset = hf[self.TEXT_CATEGORY]

            for i in range(len(cleaned_text_doc_dset)):
                cleaned_text = cleaned_text_doc_dset[i]
                doc_id = cleaned_doc_id_dset[i]
                category = category_dset[i]
                yield doc_id, cleaned_text, category

    def _get_corpus_label_stream(self):
        data = self._get_corpus()
        corpus_label_stream = map(operator.itemgetter(2), data)
        corpus_label_stream = map(self.decode_strings, corpus_label_stream)
        corpus_label_stream = map(int, corpus_label_stream)
        return corpus_label_stream

    def get_corpus_tfidf_stream(self):
        ngram_bow_stream = self._get_corpus_bow_stream()
        return self.tfidf[ngram_bow_stream]

    def _get_corpus_bow_stream(self):
        corpus_text_stream = self._get_corpus_text_stream()
        return self._cleaned_text_to_ngram_bow(corpus_text_stream)

    def data_directory(self):
        global _dir_name
        _dir_name = '/tmp/zerox'
        if _dir_name is None:
            _dir_name = tempfile.mkdtemp(prefix='zerox-')
            logging.info('Created directory: %s', _dir_name)
        return _dir_name

    @property
    def _h5py_filename(self):
        return self.tfidf_filename + '.text_model_hdf5.h5'

    @property
    def _tfidf_model_filename(self):
        return self.tfidf_filename

    @property
    def _dictionary_filename(self):
        return self.tfidf_filename + '.dictionary'

    @staticmethod
    def clean_str(_, text):
        text = text.replace(',', ' ')
        return _, text.lower().strip()

    @staticmethod
    def _clean_str(text):
        # TODO: Common interface please!!
        """
        TODO: Common interface please

        :param text: 
        :return: 
        """
        _, text = TextModel.clean_str(1, text)
        return text
