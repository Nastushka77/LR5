# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 18:16:49 2023

@author: User
"""
#!/usr/bin/python3
import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy as np; print("NumPy", np.__version__)
import tensorflow as tf; print("Tensorflow", tf.__version__)
import os
import time
from tensorflow.keras.layers.experimental import preprocessing
# Read, then decode for py2 compat.
path_to_file = tf.keras.utils.get_file(
 'shakespeare.txt',
'https://storage.googleapis.com/download.tensorflow.org/data/shakespea
re.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# The unique characters in the file
vocab = sorted(set(text))
BATCH_SIZE = 64
BUFFER_SIZE = 10000
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 256
RNN_UNITS = 1024
ids_from_chars = preprocessing.StringLookup(
        vocabulary=list(vocab), mask_token=None)
chars_from_ids = preprocessing.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(),
        invert=True,
        mask_token=None)
class GenModelv0(tf.keras.Model):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(
                vocab_size, embedding_dim, mask_zero=True)
        self.gru = tf.keras.layers.GRU(
                rnn_units,
                return_sequences=True,
                return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
    def call(self,
             inputs,
             states=None,
             return_state=False,
             training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(
            x,
            initial_state=states, 
            training=training)
        x = self.dense(x, training=training)
        
        if return_state:
            return x, states
        else:
            return x
        
    def get_config(self):
        base_config = super().get_config()
        base_config["embedding"] = self.embedding.get_config()
        base_config["gru"] = self.gru.get_config()
        base_config["dense"] = self.dense.get_config()
        base_config["vocab_size"] = self.vocab_size
        base_config["embedding_dim"] = self.embedding_dim
        base_config["rnn_units"] = self.rnn_units
        return base_config
 
 @classmethod
 def from_config(cls, config):
     cls = cls(**config)
     cls.embedding = tf.keras.layers.Embedding(
             config["vocab_size"],
             config["embedding_dim"], mask_zero=True)
     cls.gru = tf.keras.layers.GRU(
             config["rnn_units"],
             return_sequences=True,
             return_state=True)
     cls.dense = tf.keras.layers.Dense(config["vocab_size"])
     cls.vocab_size = config["vocab_size"]
     cls.embedding_dim = config["embedding_dim"]
     cls.rnn_units = config["rnn_units"]
     
     return cls
 
    def prepare_training_dataset(text:str):
        converted_string = ids_from_chars(
                tf.strings.unicode_split([text], 'UTF-8'))
        ids_dataset = tf.data.Dataset.from_tensor_slices(
                converted_string[0])
        if len(ids_dataset) >= 100:
            seq_length = 100
        else:
            seq_length = len(ids_dataset)
        sequences = ids_dataset.batch(
                seq_length+1, drop_remainder=True)
        
        def split_input_target(sequence):
            input_text = sequence[:-1]
            target_text = sequence[1:]
            return input_text, target_text
        dataset = sequences.map(split_input_target)
        
        return dataset
    
 class OneStep(tf.keras.Model):
     def __init__(self,
                  model,
                  chars_from_ids,
                  ids_from_chars,
                  temperature=1.0):
         super().__init__()
         self.temperature = temperature
         self.model = model
         self.chars_from_ids = chars_from_ids
         self.ids_from_chars = ids_from_chars
 # Create a mask to prevent "[UNK]" from being generated.
         skip_ids = self.ids_from_chars(
                 ['[UNK]'])[:, None]
         sparse_mask = tf.SparseTensor(
 # Put a -inf at each bad index.
         values=[-float('inf')]*len(skip_ids),
         indices=skip_ids,
 # Match the shape to the vocabulary
         dense_shape=[len(
                 ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(
            sparse_mask)
 @tf.function
 def generate_one_step(self, inputs, states=None):
     # Convert strings to token IDs.
     inputs = str(inputs)
     input_chars = tf.strings.unicode_split(
             inputs, 'UTF-8') 
     input_ids = self.ids_from_chars(
             input_chars)
     input_ids = tf.expand_dims(input_ids, 0)
 # Run the model.
 # predicted_logits.shape is
 # [batch, char, next_char_logits]
     predicted_logits, states = self.model(
             inputs=input_ids, states=states,
             return_state=True)
 # Only use the last prediction.
     predicted_logits = predicted_logits[:, -1, :]
     predicted_logits = predicted_logits/self.temperature
 # Apply the prediction mask: prevent "[UNK]" from being 
 # generated.
     predicted_logits = predicted_logits + self.prediction_mask
 # Sample the output logits to generate token IDs.
     predicted_ids = tf.random.categorical(
             predicted_logits, num_samples=1)
     predicted_ids = tf.squeeze(
             predicted_ids, axis=-1)
 # Convert from token ids to characters
     predicted_chars = self.chars_from_ids(
             predicted_ids)
 # Return the characters and model state.
     return predicted_chars, states
