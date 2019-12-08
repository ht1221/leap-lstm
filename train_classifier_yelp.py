# coding: utf-8
# author: Ting Huang
import tensorflow as tf
import numpy as np
import time
import os
import sys
import skiplstm
import data_help_yelp

flags = tf.app.flags
flags.DEFINE_integer('hidden_units', 300, 'hidden dim of lstm')
flags.DEFINE_integer('emb_size', 300, 'word embediding size')
flags.DEFINE_integer('depth', 1, 'depth')
flags.DEFINE_integer('voc_size', 170214, 'vocabulary size')
flags.DEFINE_integer('batch_size', 32, 'vocabulary size') 
flags.DEFINE_integer('nb_classes', 0, 'number of the classes')
flags.DEFINE_integer('nb_epoches', 5, 'number of the epoches')
flags.DEFINE_integer('check_step', 100, 'number of the steps for each check')
flags.DEFINE_integer('save_epoch', 100, 'number of the epoch for each saving model parameters')
flags.DEFINE_integer('gpu', 0, 'which gpu to use')
flags.DEFINE_integer('reload', -1, 'which step to reload')  # -1 for no reload
flags.DEFINE_integer('max_sentence_length', 200, 'max sentence length in data set to cut')
flags.DEFINE_integer('rnn_pattern', 1, 'implimentation pattern of general rnn')
flags.DEFINE_integer('decay_start', 100, 'epoch to strart learning decay')
flags.DEFINE_integer("if_schedule", 0, "if use schedule training to imutate the training process of skip-lstm")


flags.DEFINE_float('temperature', 0.1, 'temperature for gumbel-softmax estimator')
flags.DEFINE_boolean('temperature_anneal', False, 'if anneal for temperature')

flags.DEFINE_string('rnn_cell_type', 'lstm', 'number layer of rnn')
flags.DEFINE_string('yelp_set', 'FULL', 'yelp set version: FULL or BINARY')
flags.DEFINE_string('rnn_model', "leap-lstm", "rnn cell to use： skip-rnn-2017, leap-lstm, rnn")

flags.DEFINE_float("mask_rate_for_training", 0.0, "mask rate for the schedule training")
flags.DEFINE_float('gpu_rate', '0.9', 'how much gpu to use')
flags.DEFINE_float('max_gradient_norm', 5.0, 'max gradient norm for clip')
flags.DEFINE_float('keep_prob_lstm', 0.9, 'keep prob for lstm dropout')
flags.DEFINE_float('keep_prob_word', 0.8, 'keep prob for word dropout')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_float('skip_reg_weight', 0.1, 'weight for skip regularization')
flags.DEFINE_float('target_skip_rate', 0.0, 'target skip rate for ybyl-lstm training')
flags.DEFINE_float('COST_PER_SAMPLE', 1e-4, 'budget loss weight for skip-rnn-2017')


# flags.DEFINE_boolean('if_skip', True, 'if use skip lstm')
flags.DEFINE_boolean('use_lstm_dropout', True, 'if use lstm dropout during training')
flags.DEFINE_boolean('use_dropout', True, 'if use dropout during training')

FLAGS = flags.FLAGS

def main(argv):
    if FLAGS.gpu > -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    print("use gpu: ", FLAGS.gpu, " gpu rate:, ", FLAGS.gpu_rate)
    print("learning rate: ", FLAGS.learning_rate, " save epoch: ", FLAGS.save_epoch, "check_step: ", FLAGS.check_step)
    print("temperature_anneal:", FLAGS.temperature_anneal, " initial temperature: ", FLAGS.temperature)
    
    # here data path needs to be modified
    if FLAGS.yelp_set == 'BINARY':
        FLAGS.nb_classes = 2
        data_source = {'text': './data/Yelp/yelp.p', 'glove': './data/Yelp/yelp_glove.p'}
    elif FLAGS.yelp_set == 'FULL':
        FLAGS.nb_classes = 5
        data_source = {'text': './data/Yelp/yelp_full.p', 'glove': './data/Yelp/yelp_full_glove.p'}

    # load yelp(FULL/BINARY) data set
    data = data_help_yelp.YelpData(data_source=data_source, nb_classes=FLAGS.nb_classes, type=FLAGS.yelp_set, LENGTH_LIMIT=FLAGS.max_sentence_length)
    # print("train / dev / test : {}, {}, {}".format(len(data.train_labels), len(data.dev_labels), len(data.test_labels)))
    print("load data successfully")

    pre_trained_emb = data.glove_emb
    print("load word embedding successfully")

    # load model
    model = skiplstm.SkipLSTMClassifier(FLAGS, pre_trained_emb)
    print("load model successfully")

    # create session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    model_hyperparameters = "yelp_{}_lr_{}".format(FLAGS.yelp_set.lower(), FLAGS.learning_rate)
    model_name = 'skip-lstm' if FLAGS.rnn_model=='lbyl-lstm' else 'lstm'
    checkpoint_dir = "/home/ht/SkipText/save/classification/yelp_{}/".format(FLAGS.yelp_set.lower())+ model_name + '/'
    saver = tf.train.Saver()
    if FLAGS.reload >= 0:
        saver.restore(sess, save_path=checkpoint_dir + "{}.ckpt-{}".format(model_hyperparameters, FLAGS.reload))
        print("Reload successfully")
    print("Start training !")
    best_acc = [0.0, 0.0]   # 0 for dev, 1 for test

    # print("After initialization")
    # print("deve loss: {}, deve acc: {}, deve skip rate: {}".format(deve_loss, deve_acc, deve_skip_rate))
    # print("test loss: {}, test acc: {}, test skip rate: {}".format(test_loss, test_acc, test_skip_rate))
    if FLAGS.if_schedule == 1:
        FLAGS.mask_rate_for_training = 0.3
        print("mask rate for training: ", FLAGS.mask_rate_for_training)
    else:
        FLAGS.mask_rate_for_training = -10.0
    for epoch in range(FLAGS.reload + 1, FLAGS.nb_epoches):
        if epoch >= FLAGS.decay_start:
            FLAGS.learning_rate = FLAGS.learning_rate / 2.0
        print("Epoch {}, lr: {}".format(epoch, FLAGS.learning_rate))
        
        sum_train_loss = 0.0
        sum_correct_samples = 0
        nb_batches = len(data.train_labels) // FLAGS.batch_size
        data.shullfe_train_set()
        start_time = time.time()
        for step in range(nb_batches):
            print("epoch {}, step {}".format(epoch, step))
            step_loss, step_nb_right, step_skip_rate = train_step(sess, model, data, step)
            # print("loss: ", step_loss, " step_nb_right: ", step_nb_right)
            # exit()
            step_acc = step_nb_right * 1.0 / FLAGS.batch_size

            sum_train_loss += step_loss * FLAGS.batch_size
            sum_correct_samples += step_nb_right

            if (step+1) % FLAGS.check_step == 0:
                
                deve_loss, deve_acc, deve_skip_rate = evaluate(sess, model, data, type='dev')
                test_loss, test_acc, test_skip_rate = evaluate(sess, model, data, type='test')
                best_acc = update_best_acc(best_acc, [deve_acc, test_acc])

                print("on step ", step+1)
                print("train loss: ", step_loss)
                print("deve loss: {}, deve acc: {}, deve skip rate: {}".format(deve_loss, deve_acc, deve_skip_rate))
                print("test loss: {}, test acc: {}, test skip rate: {}".format(test_loss, test_acc, test_skip_rate))

                # if FLAGS.if_schedule == 1:
                #     FLAGS.mask_rate_for_training = FLAGS.mask_rate_for_training - 0.15 # 0.3, 0.15, 0.0
                #     print("mask rate for training: ", FLAGS.mask_rate_for_training)
                if FLAGS.temperature_anneal:
                    sess.run(model.assign_temperature)
                    print("Current trmperature: ", sess.run(model.temperature))
        if FLAGS.if_schedule == 1:
                FLAGS.mask_rate_for_training = FLAGS.mask_rate_for_training - 0.15 # 0.3, 0.15, 0.0
                print("mask rate for training: ", FLAGS.mask_rate_for_training)

        epoch_time = time.time() - start_time
        average_train_loss = sum_train_loss / (FLAGS.batch_size * nb_batches) 
        train_acc = sum_correct_samples * 1.0 / (FLAGS.batch_size * nb_batches)

        deve_loss, deve_acc, deve_skip_rate = evaluate(sess, model, data, type='dev')
        test_loss, test_acc, test_skip_rate = evaluate(sess, model, data, type='test')
        best_acc = update_best_acc(best_acc, [deve_acc, test_acc])
        print("This epoch consumes time {} s".format(epoch_time))
        print("After epoch {}, train acc: {}, deve acc: {} , test acc: {}".format(epoch, train_acc, deve_acc, test_acc))
        print("deve skip rate: {}, test skip rate: {}".format(deve_skip_rate, test_skip_rate))
        if (epoch+1) % FLAGS.save_epoch == 0:
            saver.save(sess, checkpoint_dir + "{}.ckpt".format(model_hyperparameters), global_step=epoch)
            print("Save model successfully.")

    print('Best test accuracy: {}'.format(best_acc[-1]))


def train_step(sess, model, data, idx):
    # prepare batch data
    batch, lengths, labels = data.get_batch(idx, FLAGS.batch_size, type='train')
    batch = mask_lstm_inputs(batch, FLAGS.mask_rate_for_training)    # shcedule-training
    masks = get_input_mask(batch)

    feed_dict = {model.sentence: batch,
                 model.sentence_mask: masks,
                 model.label: labels,
                 model.keep_lstm_prob_placeholder: FLAGS.keep_prob_lstm,
                 model.keep_word_prob_placeholder: FLAGS.keep_prob_word,
                 model.learning_rate: FLAGS.learning_rate}
    _, loss, nb_right, skip_rate = sess.run([model.train_op, model.loss, model.correct_num, model.skip_rate], feed_dict)
    # print("loss: ", loss, " nb right: ", nb_right)
    # print("sentence presentation: ", sp)
    return loss, nb_right, skip_rate


def evaluate(sess, model, data, type):
    '''
    在完整的 test/dev set上计算loss, acc
    '''
    if type == 'dev':
        data_labels = data.dev_labels
    else:
        data_labels = data.test_labels

    nb_samples = len(data_labels)
    nb_batches = nb_samples // FLAGS.batch_size
    if nb_batches * FLAGS.batch_size < nb_samples:
        nb_batches += 1
    sum_loss = 0.0
    sum_correct_samples = 0.0
    sum_nb_skip = 0.0
    sum_symbols = 0.0
    all_updated_states = []
    all_masks = []
    for step in range(nb_batches):
        batch, lengths, labels = data.get_batch(step, FLAGS.batch_size, type=type)
        masks = get_input_mask(batch)

        feed_dict = {model.sentence: batch,
                     model.sentence_mask: masks,
                     model.label: labels,
                     model.keep_lstm_prob_placeholder: 1.0,
                     model.keep_word_prob_placeholder: 1.0}
        step_loss, step_nb_right, step_nb_skip, step_nb_symbols = sess.run([model.loss, model.correct_num, model.nb_skip, model.nb_symbols], feed_dict)
        sum_loss += step_loss * len(labels)
        sum_correct_samples += step_nb_right
        sum_nb_skip += step_nb_skip
        sum_symbols += step_nb_symbols

        if FLAGS.rnn_model == "skip-rnn-2017":
            step_updated_states = sess.run([model.updated_states], feed_dict)[0]
            all_updated_states.append(step_updated_states)
            all_masks.append(masks)
            # print(step_updated_states.shape)  # (None, length, 1)
       
    average_loss = sum_loss / nb_samples
    acc = sum_correct_samples * 1.0 / nb_samples
    skip_rate = sum_nb_skip / sum_symbols

    return average_loss, acc, skip_rate

def update_best_acc(cur_best, new_acc):
    if cur_best[0] < new_acc[0]:
        return new_acc
    return cur_best

def get_input_mask(x_input, mask_id=0, dtype=np.float32):
    mask = np.not_equal(x_input, mask_id)
    mask = mask.astype(dtype)
    return mask

def mask_lstm_inputs(inputs, mask_rate):
    if mask_rate <= 0.0:
        return inputs
    nb_samples = inputs.shape[0]
    pad_length = inputs.shape[-1]
    nb_mask_token = int(mask_rate * pad_length)
    shuffled_indices = np.random.permutation(np.arange(pad_length))
    masked_inputs = []
    for line in inputs:
        np.random.shuffle(shuffled_indices)
        indices = shuffled_indices[:nb_mask_token]
        for idx in indices:
            line[idx] = 0.0
        masked_inputs.append(line)
    masked_inputs = np.asarray(masked_inputs, dtype=np.float32)

    return masked_inputs

if __name__ == "__main__":
    tf.app.run()
