# coding: utf-8
# author: Ting Huang
import pickle as pkl
import numpy as np

class YelpData(object):
    def __init__(self, data_source, nb_classes, batch_size=128, LENGTH_LIMIT=200, type='FULL'):
        self.data_source = data_source  # dict:{NAME: PATH}
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.LENGTH_LIMIT = LENGTH_LIMIT
        self.data_set = type    # 'FULL' for yelp full, 'BINARY' for yelp binary
        self.train_idx_seqs, self.dev_idx_seqs, self.test_idx_seqs, self.train_labels, self.dev_labels,\
        self.test_labels, self.voc, self.rev_voc, self.glove_emb = self.load_data()
        if self.data_set == 'FULL': 
            self.train_labels = self.onehot2label(self.train_labels)
            self.dev_labels = self.onehot2label(self.dev_labels)
            self.test_labels = self.onehot2label(self.test_labels)

        print("train/dev/test: {} {} {}".format(len(self.train_idx_seqs), len(self.dev_idx_seqs), len(self.test_idx_seqs)))
        print("nb_classes: ", self.train_labels.max()+1)
        print(self.dev_labels[:100])
    def load_data(self):
        # train_text val_text, test_text, train_label, val_label, test_label, dictionary and reverse_dictionary
        text_path = self.data_source['text']
        glove_path = self.data_source['glove']

        with open(text_path, 'rb') as t_f:
            if self.data_set == 'FULL':
                text_data = pkl.load(t_f, encoding='iso-8859-1')
            else:
                text_data = pkl.load(t_f)
        with open(glove_path, 'rb') as g_f:
            glove_data = pkl.load(g_f, encoding='iso-8859-1')

        return np.asarray(text_data[0]), np.asarray(text_data[1]), np.asarray(text_data[2]), np.asarray(text_data[3]), \
               np.asarray(text_data[4]), np.asarray(text_data[5]), text_data[6], text_data[7], glove_data

    def get_batch(self, batch_idx, batch_size=None, type='train'):

        if batch_size is None:
            batch_size = self.batch_size
        if type == 'train':
            data_seqs = self.train_idx_seqs
            data_labels = self.train_labels
        elif type == 'dev':
            data_seqs = self.dev_idx_seqs
            data_labels = self.dev_labels
        elif type ==  'test':
            data_seqs = self.test_idx_seqs
            data_labels = self.test_labels
        else:
            print("Wrong batch type")
            exit()

        data_size = len(data_seqs)
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, data_size)
        row_batch = data_seqs[start_idx:end_idx]

        # padding
        batch_lengths = [min(len(sample), self.LENGTH_LIMIT) for sample in row_batch]
        pad_batch = np.zeros((len(row_batch), self.LENGTH_LIMIT), dtype="int32")
        for i, sample in enumerate(row_batch):
            pad_batch[i, :batch_lengths[i]] = sample[:batch_lengths[i]]

        batch_labels = data_labels[start_idx:end_idx]

        return pad_batch, batch_lengths, batch_labels

    def shullfe_train_set(self):
        np.random.seed(241221)
        data_size = len(self.train_labels)
        shuffle_indices = np.random.permutation(np.arange(data_size))
        self.train_idx_seqs = self.train_idx_seqs[shuffle_indices]
        self.train_labels = self.train_labels[shuffle_indices]

    def onehot2label(self, onehot):
        # onehot: (nb_samples, nb_classes, 1)
        LABEL_IDX = np.array(range(self.nb_classes), dtype=np.int32)
        data_size = len(onehot)
        onehot = np.reshape(onehot, [data_size, self.nb_classes])
        labels = []
        for item in onehot:
            label = (item * LABEL_IDX).sum()
            labels.append(label)
        labels = np.array(labels, dtype=np.int32)

        return labels



if __name__ == '__main__':
    # data path needs to be modified
    yelp = YelpData(data_source={'text': './data/Yelp/yelp_full.p', 'glove': './data/Yelp/yelp_full_glove.p'},
                         nb_classes=5,
                         type='FULL')
    print("load yelp full successfully")

    seqs = np.concatenate([yelp.train_idx_seqs, yelp.dev_idx_seqs, yelp.test_idx_seqs], axis=0)
    lengths = [len(sample) for sample in seqs]
    lengths = np.array(lengths, dtype=np.int32)
    
    print("average length: ", lengths.mean())
    print("min length: ", lengths.min())
    print("max length: ", lengths.max())
