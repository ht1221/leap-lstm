# coding: utf-8
# author: Ting Huang

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.python.ops.rnn_cell import LSTMCell, GRUCell, DropoutWrapper
from tensorflow.python.layers.core import Dense
import time
import math
import sys
# sys.path.insert(0, "../src/")
sys.path.insert(0, "../")
sys.path.insert(0, "./src/")

# skip_rnn is the cell inmplementation of skip-lstm(2017)
from rnn_cells.skip_rnn_cells import SkipLSTMCell as skip_rnn


def sample_gumbel(shape, eps=1e-20): 
		"""Sample from Gumbel(0, 1)"""
		U = tf.random_uniform(shape,minval=0,maxval=1)
		return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature=1e-5): 
	""" Draw a sample from the Gumbel-Softmax distribution"""
	y = logits + sample_gumbel(tf.shape(logits))
	return tf.nn.softmax(y / temperature)

def xavier_weights_variable(shape, name_prefix):
	if len(shape) == 4:
		scale = np.sqrt(6.0 / (shape[0]*shape[1]*shape[2] + shape[-1]))
	else:
		scale = np.sqrt(6.0 / (shape[0] + shape[1]))
	initial = tf.random_uniform(shape, -scale, scale, dtype=tf.float32)
	weights = tf.get_variable(name_prefix + "_w", initializer=initial)
	return weights

def bias_variable(shape, name_prefix):
	initial = tf.constant(0.0001, shape=shape)
	return tf.get_variable(name_prefix + "_b", initializer=initial)

def word_level_cnn_no_pooling(inputs, sentence_length, kernel_size_list, keep_prob=None, activation=tf.nn.relu):
	'''
	:param inputs:  shape: (nb_samples, sentence_length, dim_word_vector, 1)
	:param sentence_length: sentence lengths
	:param kernel_size_list: filter size list
	:param keep_prob: dropout for cnn
	:param activation:
	:return: cnn features and parameters (w+b)
	'''
	features = []
	w = []
	b = []
	feature_size = 0
	for kernel_size in kernel_size_list:
		print("kernel size is: ", kernel_size)
		feature_size += kernel_size[-1]
		conv_w = xavier_weights_variable(shape=kernel_size, name_prefix="sentence_cnn_" + str(kernel_size[0]))
		conv_b = bias_variable(shape=[kernel_size[-1], ], name_prefix="sentence_cnn_" + str(kernel_size[0]))
		conv_out = tf.nn.conv2d(inputs, conv_w, padding="SAME", strides=[1, 1, 300, 1])
		conv_out = activation(conv_out + conv_b)
		w.append(conv_w)
		b.append(conv_b)
		features.append(tf.reshape(conv_out, shape=[-1, sentence_length, kernel_size[-1]]))
		# shape: (nb_samples, length, 1, 100)
		# outputs = tf.nn.dropout(max_pooling_out, keep_prob=keep_prob)
		# return conv_w, conv_b, outputs
	print("feature_size is: ", feature_size)
	sequence_features = tf.concat(features, axis=-1)
	# shape: (nb_samples, sentence_length, 300)
	return w, b, sequence_features, feature_size


class SkipLSTMClassifier(object):
	# "SKipLSTM" in "SkipLSTMClassifier" means LSTM cell that can skip words, not the skip-lstm (2017).
	# And skip-lstm (2017) is one of the "SKipLSTMs"
	def __init__(self, config, pre_trained_word_emb=None, is_training=True):
		self.COST_PER_SAMPLE =  1e-4                        # used for skip-lstm (2017)
		self.target_skip_rate = config.target_skip_rate     # gamma
		self.config = config
		self.depth = config.depth   # only one SkipLSTM layer used 
		self.max_sentence_length = config.max_sentence_length
		self.rnn_cell_type = config.rnn_cell_type
		self.hidden_units = config.hidden_units
		self.nb_classes = config.nb_classes
		self.voc_size = config.voc_size
		self.emb_size = config.emb_size
		self.use_lstm_dropout = config.use_lstm_dropout
		self.max_gradient_norm = config.max_gradient_norm
		self.temperature = config.temperature
		self.pre_trained_word_emb = pre_trained_word_emb
		self.rnn_model = config.rnn_model
		self.rnn_pattern = config.rnn_pattern
		self.skip_reg_weight = config.skip_reg_weight
		self.dtype = tf.float32
		self.is_training = is_training
		self.build_model()
		
	def build_model(self):
		self.init_placeholders()
		self.s2v()  # sentence to vector(sentence representation)
		self.pred_layer()

	def init_placeholders(self):
		self.sentence = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_sentences")
		self.sentence_mask = tf.placeholder(tf.float32, [None, None], name='sentence_mask')
		self.label = tf.placeholder(dtype=tf.int32, shape=[None, ], name='gt_label')
		self.keep_lstm_prob_placeholder = tf.placeholder(self.dtype, name='keep_lstm_prob')
		self.keep_word_prob_placeholder = tf.placeholder(self.dtype, name='keep_word_prob')
		self.learning_rate = tf.placeholder(tf.float32, name="lr")
		self.batch_size = tf.shape(self.sentence)[0]

	def s2v(self):
		sqrt3 = math.sqrt(3.0)
		initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=self.dtype)

		# word embedding layer
		if self.pre_trained_word_emb is not None:
			self.word_embeddings = tf.get_variable(name='word_embedding',
												   initializer=self.pre_trained_word_emb,
												   dtype=self.dtype)
		else:
			self.word_embeddings = tf.get_variable(name='word_embedding',
												   shape=[self.voc_size, self.emb_size],
												   initializer=initializer,
												   dtype=self.dtype)
		self.embedded_sentence = tf.nn.embedding_lookup(self.word_embeddings, self.sentence)
		self.embedded_sentence = tf.nn.dropout(self.embedded_sentence, keep_prob=self.keep_word_prob_placeholder)
		
		# create the rnn cell
		if self.rnn_cell_type.lower() == 'gru':
			rnn_cell = GRUCell
		else:
			rnn_cell = LSTMCell
		rnn_cell = rnn_cell(self.hidden_units)
 
		if self.use_lstm_dropout:
			rnn_cell = DropoutWrapper(rnn_cell, dtype=tf.float32, output_keep_prob=self.keep_lstm_prob_placeholder)
		if self.rnn_model == 'leap-lstm':
			self.sentence_emb, self.skip_dis_output = self.leap_lstm(rnn_cell)
		elif self.rnn_model == 'rnn':
			if self.rnn_pattern == 1:
				self.sentence_emb = self.general_rnn(rnn_cell, out='LAST')
			else:
				self.sentence_emb = self.general_rnn_for_pattern(rnn_cell, out='LAST')  # for test the training time
		elif self.rnn_model == 'brnn':
			self.sentence_emb = self.general_brnn()
		elif self.rnn_model == 'skip-rnn-2017':
			self.sentence_emb, self.budget_loss, self.updated_states, self.rnn_final_states, self.rnn_outputs = self.skip_rnn_2017()
		elif self.rnn_model == 'skim-rnn':
			small_rnn_cell = LSTMCell(5)    # small size 5
			small_rnn_cell = DropoutWrapper(small_rnn_cell, dtype=tf.float32, output_keep_prob=self.keep_lstm_prob_placeholder)
			self.sentence_emb, self.skip_dis_output, self.skim_loss = self.skim_rnn(rnn_cell, small_rnn_cell)   # skim-rnn的设定直接按照github上源码来就可以了
		else:
			print("bad rnn model!")
			exit()

	def skim_rnn(self, large_rnn_cell, small_rnn_cell):
		''' reproduction of tf-version skim-rnn (2017), another related search '''
		initial_state = large_rnn_cell.zero_state(self.batch_size, dtype=self.dtype)  # ([None, hidden_units], [None, hidden_units])
		state = initial_state
		with tf.variable_scope('skim_rnn') as scope:
			pred_skip_dense = Dense(units=2,
									kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
									activation=None)
			skip_dis_set = []
			outputs = []
			for step in range(self.max_sentence_length):
				if step > 0:
					tf.get_variable_scope().reuse_variables()
				x_t = self.embedded_sentence[:, step, :]    # (None, emb_size)

				# compute skip rate for this step
				skip_pred_logits = pred_skip_dense(tf.concat([state[0], state[1], x_t], axis=-1))   # using (h,c, x) to compute the skip probability distribution
				# shape: (None, 2)

				# using umbel-softmax to sample from the bernoulli distribution to decide to skip
				skip_dis = gumbel_softmax_sample(logits=skip_pred_logits, temperature=0.5)  # default temperature 0.5 in skim-rnn
				skip_dis_set.append(skip_dis)

				# process large rnn and small rnn
				small_state = (state[0][:, :5], state[1][:, :5])            # shape: ([None, 5], [None, 5])
				_, step_large_state = large_rnn_cell(x_t, state)            # shape: ([None, 100], [None, 100])
				_, step_small_state = small_rnn_cell(x_t, small_state)

				# update state
				tmp_skip_dis = tf.stack([skip_dis]*2)  # shape: (2, None, 2)

				state_for_skim = tf.concat([tf.stack(step_small_state, axis=0), tf.stack(state, axis=0)[:, :, 5:]], axis=-1)    # (2, None, 100)
				state_for_keep = step_large_state
				new_state = state_for_skim * tf.expand_dims(tmp_skip_dis[:, :, 1], -1) + state_for_keep * tf.expand_dims(tmp_skip_dis[:, :, 0], -1)
				# 1 for skim, 0 for keep

				# operations about mask
				step_mask = tf.expand_dims(tf.stack([self.sentence_mask[:, step]]*2), -1)  # (2, None, 1)
				new_state = step_mask * new_state + (1.0 - step_mask) * state
				new_output = new_state[-1]
				outputs.append(new_output)

				# update the state in the last
				state = tf.unstack(new_state)

			skip_dis_output = tf.stack(skip_dis_set, axis=1)    # shape: (None, max_sentence_length, 2)

			# loss for skimming prob
			# L ^ { \prime } ( \theta ) = L ( \theta ) + \gamma \frac { 1 } { T } \sum _ { t } - \log \left( \mathbf { p } _ { t } ^ { 2 } \right)
			prob_skim = skip_dis_output[:, :, 1]
			negative_los_skim_prob = -tf.log(tf.pow(prob_skim, 2.0)+1e-10)  # (None, max_sentence_length)
			average_weight = tf.nn.softmax(self.sentence_mask * (self.sentence_mask + 5.0))

			gamma = 0.00015
			skim_loss = gamma * tf.reduce_mean(tf.reduce_sum(average_weight * negative_los_skim_prob, axis=-1))

			return outputs[-1], skip_dis_output, skim_loss

	def leap_lstm(self, rnn_cell):
		if self.rnn_cell_type.lower() == 'gru':
			rnn_cell_b = GRUCell
		else:
			rnn_cell_b = rnn.LSTMCell
		small_cell_size = 10
		rnn_cell_reverse = rnn_cell_b(small_cell_size)

		if self.use_lstm_dropout:
			rnn_cell_reverse = DropoutWrapper(rnn_cell_reverse, dtype=tf.float32, output_keep_prob=self.keep_lstm_prob_placeholder)
		
		back_rnn_h = self.general_rnn_reversed(rnn_cell_reverse)
		cnn_inputs = tf.reshape(self.embedded_sentence, shape=[-1, self.max_sentence_length, self.emb_size, 1])
		_1, _2, cnn_features, self.cnn_f_size = word_level_cnn_no_pooling(cnn_inputs, self.max_sentence_length, kernel_size_list=[(3, self.emb_size, 1, 60), (4, self.emb_size, 1, 60), (5, self.emb_size, 1, 60)])

		initial_state = rnn_cell.zero_state(self.batch_size, dtype=self.dtype)  # ([None, hidden_units], [None, hidden_units])
		state = initial_state
		with tf.variable_scope('leap_lstm') as scope:
			leap_dense_1 = Dense(units=100,
								 kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
								 activation=tf.nn.relu)
			leap_dense_2 = Dense(units=2,
								 kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

			skip_dis_set = []
			outputs = []
			P_set = []
			Q_set = []
			for step in range(self.max_sentence_length):
				if step > 0:
					tf.get_variable_scope().reuse_variables()
				x_t = self.embedded_sentence[:, step, :]    # (None, emb_size)
				cnn_f_t = cnn_features[:, step, :]
				if step == self.max_sentence_length-1:
					rnn_reverse_h_t = tf.zeros(shape=[self.batch_size, small_cell_size])
				else:
					rnn_reverse_h_t = back_rnn_h[:, step+1, :]  # back rnn中存储的后文信息
				
				# 计算这一步skip的概率
				# skip_h1 = pred_skip_dense_1(tf.concat([state[-1], x_t], axis=-1))   # 上一步的h和这一步的x来确定是否需要skip
				# # (None, hidden_units+emb_dim) --> (None, 100)
				# skip_pred_logits = pred_skip_dense_2(skip_h1)
				# shape: (None, 2), 这一步的skip的概率分布的logits


				concatted_input = tf.concat([state[-1], x_t, rnn_reverse_h_t, cnn_f_t], axis=-1)
				concatted_input = tf.reshape(concatted_input, shape=[-1, self.hidden_units+self.emb_size+small_cell_size+self.cnn_f_size])
				h_1 =  leap_dense_1(concatted_input)
				skip_pred_logits= leap_dense_2(h_1)
				skip_probability = tf.nn.softmax(skip_pred_logits)

				skip_dis = gumbel_softmax_sample(logits=skip_pred_logits, temperature=self.temperature)   # (None, 2)
				skip_dis_set.append(skip_dis)


				# compute one-step rnn process
				_, step_state = rnn_cell(x_t, state)

				tmp_skip_dis = tf.stack([skip_dis]*2)  #(2, None, 2)
				new_state = step_state * tf.expand_dims(tmp_skip_dis[:, :, 0], -1) + state * tf.expand_dims(tmp_skip_dis[:, :, 1], -1)
				# 1 for skip, 0 for keep

				# operations about mask
				step_mask = tf.expand_dims(tf.stack([self.sentence_mask[:, step]]*2), -1)  # (2, None, 1)
				new_state = step_mask * new_state + (1.0 - step_mask) * state
				new_output = new_state[-1] 
				outputs.append(new_output)

				# update the state in the last
				state = tf.unstack(new_state)

			skip_dis_output = tf.stack(skip_dis_set, axis=1)    # shape (None, max_sentence_length, 2)

			return outputs[-1], skip_dis_output

	def general_rnn(self, rnn_cell, out='LAST'):
		if out == 'LAST':
			sentence_lengths = tf.reduce_sum(self.sentence_mask, axis=-1)
			initial_state = rnn_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
			_, last_state_tuple = tf.nn.dynamic_rnn(rnn_cell,
													self.embedded_sentence,
													sequence_length=sentence_lengths,
													initial_state=initial_state,
													dtype=tf.float32)
			final_output = last_state_tuple[-1] # state tuple (c, h)
		else:
			outputs, _ = tf.nn.dynamic_rnn(rnn_cell, self.embedded_sentence, dtype=tf.float32)
			# (None, sequence_length, hidden_size)

			sequence_weight = tf.nn.softmax((self.sentence_mask + 10.0) * self.sentence_mask)
			# (None, sequence_length)

			final_output = tf.reduce_sum(outputs * tf.reshape(sequence_weight,[-1, self.max_sentence_length, 1]),
										 axis=1)
		return final_output

	def general_brnn(self):
		if self.rnn_cell_type.lower() == 'gru':
			cell_type = GRUCell
		else:
			cell_type = LSTMCell
		rnn_cell_forward = cell_type(self.hidden_units/2)
		rnn_cell_backward = cell_type(self.hidden_units/2)
		if self.use_lstm_dropout:
			rnn_cell_forward = DropoutWrapper(rnn_cell_forward, dtype=tf.float32, output_keep_prob=self.keep_lstm_prob_placeholder)
			rnn_cell_backward = DropoutWrapper(rnn_cell_backward, dtype=tf.float32, output_keep_prob=self.keep_lstm_prob_placeholder)
		sentence_lengths = tf.cast(tf.reduce_sum(self.sentence_mask, axis=-1), dtype=tf.int32)
		initial_state_forward = rnn_cell_forward.zero_state(batch_size=self.batch_size, dtype=tf.float32)
		initial_state_backward = rnn_cell_backward.zero_state(batch_size=self.batch_size, dtype=tf.float32)
		_, last_state_tuple = tf.nn.bidirectional_dynamic_rnn(rnn_cell_forward,
															  rnn_cell_backward,
															  self.embedded_sentence,
															  sequence_length=sentence_lengths,
															  initial_state_fw=initial_state_forward,
															  initial_state_bw=initial_state_backward,
															  dtype=tf.float32)
		'''
		A tuple (outputs, output_states) where: * outputs: A tuple (output_fw, output_bw) containing the
		forward and the backward rnn output Tensor. If time_major == False (default), output_fw will be a Tensor shaped:
		[batch_size, max_time, cell_fw.output_size] and output_bw will be a Tensor shaped:
		[batch_size, max_time, cell_bw.output_size]. If time_major == True, output_fw will be a Tensor shaped:
		[max_time, batch_size, cell_fw.output_size] and output_bw will be a Tensor shaped:
		[max_time, batch_size, cell_bw.output_size]. It returns a tuple instead of a single concatenated Tensor,
		unlike in the bidirectional_rnn. If the concatenated one is preferred, the forward and backward outputs can be
		concatenated as tf.concat(outputs, 2). * output_states: A tuple (output_state_fw, output_state_bw) containing the
		forward and the backward final states of bidirectional rnn.
		'''
		final_output = tf.concat([last_state_tuple[0][-1], last_state_tuple[1][-1]], -1)  # (None, self.hidden_units)
		return final_output

	def general_rnn_for_pattern(self, rnn_cell, out='LAST'):
		# general rnn which is implementationed by "for"
		with tf.variable_scope('general_rnn') as scope:
			initial_state = rnn_cell.zero_state(self.batch_size, dtype=self.dtype)  # ([None, hidden_units], [None, hidden_units])
			state = initial_state
			outputs = []
			for step in range(self.max_sentence_length):
				if step > 0:
					tf.get_variable_scope().reuse_variables()
				x_t = self.embedded_sentence[:, step, :]
				output, state = rnn_cell(x_t, state)
				outputs.append(output)
			# conduct the mask operator
			valid_length = tf.reduce_sum(self.sentence_mask, axis=-1)  # (None, )
			valid_length = valid_length - tf.constant(1.0, dtype=tf.float32)
			valid_length = tf.cast(valid_length, tf.int32)
			one_hot_length = tf.one_hot(valid_length, axis=-1, depth=self.max_sentence_length)
			outputs = tf.stack(outputs, axis=1) # (None, length, hidden_units)
			final_output = tf.reduce_sum(outputs * tf.reshape(one_hot_length, [-1, self.max_sentence_length, 1]), axis=1)

		return final_output

	def general_rnn_reversed(self, rnn_cell):
		'''  reversed rnn, return feature sequence'''
		with tf.variable_scope('general_rnn_reversed') as scope:
			sentence_lengths = tf.reduce_sum(self.sentence_mask, axis=-1)
			initial_state = rnn_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
			# reverse
			reversed_seq = tf.reverse_sequence(self.embedded_sentence, tf.cast(sentence_lengths, dtype=tf.int32), seq_axis=1)
			outputs, last_state_tuple = tf.nn.dynamic_rnn(rnn_cell,
														  reversed_seq,
														  sequence_length=sentence_lengths,
														  initial_state=initial_state,
														  dtype=tf.float32)

			outputs = tf.reverse_sequence(outputs, tf.cast(sentence_lengths, dtype=tf.int32), seq_axis=1)
			# (None, max_length, dim)

		return outputs

	def skip_rnn_2017(self):

		# Create SkipLSTM and trainable initial state
		cell = skip_rnn(self.hidden_units)
		# if self.use_lstm_dropout:
			# cell = DropoutWrapper(cell, dtype=tf.float32, output_keep_prob=self.keep_lstm_prob_placeholder)
		initial_state = cell.trainable_initial_state(self.batch_size)

		# Dynamic RNN unfolding
		rnn_outputs, rnn_final_states = tf.nn.dynamic_rnn(cell, self.embedded_sentence, dtype=tf.float32, initial_state=initial_state)
		
		# Split the output into the actual RNN output and the state update gate
		updated_states = rnn_outputs.state_gate

		# get the final hidden states
		final_output = rnn_outputs.h[:, -1, :]
		final_output = tf.reshape(final_output, shape=[-1, self.hidden_units])


		# Add a penalization for each state update (i.e. used sample)
		budget_loss = tf.reduce_mean(tf.reduce_sum(self.COST_PER_SAMPLE * updated_states, 1), 0)

		return final_output, budget_loss, updated_states, rnn_final_states, rnn_outputs
		
	def pred_layer(self):
		output_layer1 = Dense(units=512, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), activation=tf.nn.relu)
		output_layer2 = Dense(units=self.nb_classes, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

		output_h1 = output_layer1(self.sentence_emb)
		self.logits = output_layer2(output_h1)
		self.prob = tf.nn.softmax(self.logits)

		self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
		self.loss = tf.reduce_mean(self.loss)

		if self.rnn_model == 'leap-lstm':
			self.skip_loss, self.skip_rate = self.skip_regularization(self.skip_dis_output, self.sentence_mask, lambda_=self.skip_reg_weight) # 对skip rate进行约束
			self.cost = self.loss + self.skip_loss
			self.updated_states = tf.constant(0.0, dtype=tf.float32)	# only used for skip-lstm (2017)
		elif self.rnn_model == "skim-rnn":
			self.skip_loss, self.skip_rate = self.skip_regularization(self.skip_dis_output, self.sentence_mask, lambda_=self.skip_reg_weight)
			self.cost = self.loss + self.skim_loss
			self.updated_states = tf.constant(0.0, dtype=tf.float32)
		else:
			self.skip_loss = tf.constant(0.0, dtype=tf.float32)
			self.skip_rate = tf.constant(0.0, dtype=tf.float32)
			self.nb_skip = tf.constant(0.0, dtype=tf.float32)
			self.nb_symbols = tf.constant(1.0, dtype=tf.float32)
			self.cost = self.loss

			if self.rnn_model == 'skip-rnn-2017':
				self.cost = self.cost + self.budget_loss
			else:
				self.updated_states = tf.constant(0.0, dtype=tf.float32)
		
		self.prediction = tf.argmax(self.prob, axis=-1, output_type=tf.int32)
		self.correct_num = tf.reduce_sum(tf.cast(tf.equal(self.prediction, self.label), tf.int32))
	
		optimizer = tf.train.AdamOptimizer(self.learning_rate)
		trainable_params = tf.trainable_variables()
		gradients = tf.gradients(self.cost, trainable_params)
		clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
		self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))
	
	def skip_regularization(self, skips, mask, lambda_=0.1):
		print("Target skip rate: ", self.target_skip_rate)
		skips = tf.reshape(skips[:, :, 1], [-1, self.max_sentence_length])
		self.nb_skip = tf.reduce_sum(skips*mask)     # (None, )
		self.nb_symbols = tf.reduce_sum(mask)   # (None, )
		skip_rate = tf.divide(self.nb_skip, self.nb_symbols)

		return lambda_ * (skip_rate-self.target_skip_rate)**2, skip_rate







