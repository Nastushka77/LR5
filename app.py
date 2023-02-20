# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 18:16:49 2023

@author: User
"""
import json
import os
from flask import Flask, request
from train import prepare_training_dataset
from train import generate_text
from train import get_model
MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
print("Loading model from: {}".format(MODEL_PATH))
model = get_model(MODEL_PATH)
app = Flask(__name__)
@app.route('/line/<int:Line>')
def line(Line):
    with open('./test.txt', 'rt') as file:
        file_data = file.read()
    return json.dumps(file_data[Line])
@app.route('/prediction/', methods=['POST', 'GET'])
def prediction():
    query_string = [str(request.args.get('query'))]
    try: 
        n_characters = int(request.args.get('length'))
    except:
        n_characters = 100
    prediction = generate_text(
            model,
            n_characters=n_characters,
            query=query_string)
    return prediction
@app.route('/score', methods=['POST', 'GET'])
def score():
    with open('./test.txt', 'rt') as file:
        data_test = file.read()
    data = prepare_training_dataset(data_test)
    result = model.evaluate(data)
 
    return dict(zip(model.metrics_names, result))
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')

def generate_text(model,
                  n_characters=1000,
                  query='ROMEO:',
                  chars_from_ids=chars_from_ids,
                  ids_from_chars=ids_from_chars):
    start = time.time()
    states = None
    next_char = query
    result = [next_char]
    one_step_model = OneStep(model,
                             chars_from_ids, ids_from_chars)
    for n in range(n_characters):
        next_char, states = one_step_model.generate_one_step(
                next_char, states=states)
        result.append(next_char)
        
    result = tf.strings.join(result)
    end = time.time()
    
    return {'text':result[0].numpy().decode('utf-8'),
            'runtime': end - start}
def get_model(file_path):
    model = GenModelv0(
 # Be sure the vocabulary size matches the `StringLookup` layers.
         vocab_size=len(ids_from_chars.get_vocabulary()),
         embedding_dim=EMBEDDING_DIM,
         rnn_units=RNN_UNITS)
    loss
tf.losses.SparseCategoricalCrossentropy(from_logits=True)
     metrics = [
         tf.metrics.sparse_categorical_accuracy
     ]
     model.compile(
             optimizer='adam',
             loss=loss,
             metrics=metrics)
     model.load_weights(file_path)
     return model
 
 def train(epochs=20):
     
     train_text = text[:int(len(text)*0.8)]
     with open("train.txt", 'wb') as train:
         train.write(train_text.encode(encoding='utf-8'))
    test_text = text[:-int(len(text)*0.2)]
    with open("test.txt", 'wb') as test:
        test.write(test_text.encode(encoding='utf-8'))
    dataset = prepare_training_dataset(train_text)
    test_dataset = prepare_training_dataset(test_text)
    dataset = (
            dataset
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE))
    test_dataset = (
            test_dataset
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE))
    
 model = GenModelv0(
 # Be sure the vocabulary size matches
     the `StringLookup` layers.
     vocab_size=len(ids_from_chars.get_vocabulary()),
     embedding_dim=EMBEDDING_DIM,
     rnn_units=RNN_UNITS)
 loss = tf.losses.SparseCategoricalCrossentropy(
         from_logits=True)
     metrics = [
         tf.metrics.sparse_categorical_accuracy
    ]
    model.compile(
            optimizer='adam',
            loss=loss,
            metrics=metrics)
 # Directory where the checkpoints will be saved
     checkpoint_dir = './training_checkpoints'
 # Name of the checkpoint files
     checkpoint_prefix = os.path.join(
             checkpoint_dir, "ckpt_{epoch}")
     checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
             filepath=checkpoint_prefix,
             save_weights_only=True)
     history = model.fit(
             dataset,
             epochs=epochs,
             callbacks=[checkpoint_callback],
             validation_data=test_dataset)
     model.save_weights(os.path.join(os.environ["MODEL_DIR"],
                os.environ["MODEL_FILE"]))
     
 if __name__ == '__main__':
     if len(sys.argv) > 1:
         epochs = sys.argv[1] if len(sys.argv) >= 1 else 20
         if not isinstance(epochs, int):
             epochs = 20
         else:
             epochs = 1
        train(epochs)
$ python train.py 100
