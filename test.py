import pandas as pd
import json
import pickle
from keras import layers, activations, models, preprocessing, optimizers
from keras.utils import to_categorical
#from gensim.models import Word2Vec
import re
import numpy as np
import os

X = []
Y = []

with open("Dataset/covid-19.json") as f:
    json_data = json.load(f)

json_data = json_data['intents']
for i in range(len(json_data)):
    question = json_data[i]['patterns']
    answer = json_data[i]['responses']
    for i in range(len(question)):
        for j in range(len(answer)):
            X.append(question[i])
            Y.append(answer[j])


tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts( X + Y )
VOCAB_SIZE = len( tokenizer.word_index )+1

vocab = []
for word in tokenizer.word_index:
    vocab.append(word)

def tokenize(sentences):
    tokens_list = []
    vocabulary = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub('[^a-zA-Z0-9]', ' ', sentence)
        tokens = sentence.split()
        vocabulary += tokens
        tokens_list.append(tokens)
    return tokens_list, vocabulary

#encoder_input_data
tokenized_questions = tokenizer.texts_to_sequences(X)
maxlen_questions = max( [len(x) for x in tokenized_questions ] )
padded_questions = preprocessing.sequence.pad_sequences( tokenized_questions, maxlen = maxlen_questions, padding = 'post')
encoder_input_data = np.array(padded_questions)
print(encoder_input_data.shape, maxlen_questions)

# decoder_input_data
tokenized_answers = tokenizer.texts_to_sequences(Y)
maxlen_answers = max( [ len(x) for x in tokenized_answers ] )
padded_answers = preprocessing.sequence.pad_sequences( tokenized_answers , maxlen=maxlen_answers , padding='post' )
decoder_input_data = np.array( padded_answers )
print( decoder_input_data.shape , maxlen_answers )


# decoder_output_data
tokenized_answers = tokenizer.texts_to_sequences(Y)
for i in range(len(tokenized_answers)) :
    tokenized_answers[i] = tokenized_answers[i][1:]
padded_answers = preprocessing.sequence.pad_sequences( tokenized_answers , maxlen=maxlen_answers , padding='post' )
onehot_answers = to_categorical( padded_answers , VOCAB_SIZE )
decoder_output_data = np.array( onehot_answers )
print( decoder_output_data.shape )


encoder_inputs = layers.Input(shape=( maxlen_questions , ))
encoder_embedding = layers.Embedding( VOCAB_SIZE, 200 , mask_zero=True ) (encoder_inputs)
encoder_outputs , state_h , state_c = layers.LSTM( 200 , return_state=True )( encoder_embedding )
encoder_states = [ state_h , state_c ]

decoder_inputs = layers.Input(shape=( maxlen_answers ,  ))
decoder_embedding = layers.Embedding( VOCAB_SIZE, 200 , mask_zero=True) (decoder_inputs)
decoder_lstm = layers.LSTM( 200 , return_state=True , return_sequences=True )
decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )
decoder_dense = layers.Dense( VOCAB_SIZE , activation='softmax') 
output = decoder_dense ( decoder_outputs )

model = models.Model([encoder_inputs, decoder_inputs], output )
model.compile(optimizer=optimizers.RMSprop(), loss='categorical_crossentropy', metrics = ['accuracy'])

if os.path.exists("model/model.h5") == False:
    hist = model.fit([encoder_input_data , decoder_input_data], decoder_output_data, batch_size=50, epochs=150) 
    model.save('model/model.h5')
    f = open('model/lstm_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    model.load_weights('model/model.h5')

def make_inference_models():
    encoder_model = models.Model(encoder_inputs, encoder_states)
    decoder_state_input_h = layers.Input(shape=( 200 ,))
    decoder_state_input_c = layers.Input(shape=( 200 ,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding , initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = models.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)    
    return encoder_model , decoder_model    

def str_to_tokens( sentence : str ):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        tokens_list.append( tokenizer.word_index[ word ] ) 
    return preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=maxlen_questions , padding='post')


enc_model , dec_model = make_inference_models()
for i in range(len(X)):
    states_values = enc_model.predict( str_to_tokens( X[i] ) )
    empty_target_seq = np.zeros( ( 1 , 1 ) )
    empty_target_seq[0, 0] = tokenizer.word_index['wearing']
    stop_condition = False
    decoded_translation = ''
    while not stop_condition :
        dec_outputs , h , c = dec_model.predict(empty_target_seq + states_values )
        sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
        sampled_word = None
        for word , index in tokenizer.word_index.items() :
            if sampled_word_index == index :
                decoded_translation += ' {}'.format( word )
                sampled_word = word
        
        if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
            stop_condition = True
            
        empty_target_seq = np.zeros( ( 1 , 1 ) )  
        empty_target_seq[ 0 , 0 ] = sampled_word_index
        states_values = [ h , c ] 
    print(X[i])
    print( decoded_translation )
    print()

    


















