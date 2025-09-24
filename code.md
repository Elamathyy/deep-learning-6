import numpy as np
import nltk
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.utils import pad_sequences

# Sample data
input_texts = ['I love NLP', 'He plays football']
target_texts = [['PRON', 'VERB', 'NOUN'], ['PRON', 'VERB', 'NOUN']]

# Preprocessing
word_vocab = sorted(list(set(word for sent in input_texts for word in sent.split())))
tag_vocab = sorted(list(set(tag for tags in target_texts for tag in tags)))

# Add special tokens for padding
word_vocab.insert(0, "<PAD>")
tag_vocab.insert(0, "<PAD>")
tag_vocab.insert(1, "<START>")
tag_vocab.append("<END>")

word2idx = {word: i for i, word in enumerate(word_vocab)}
idx2word = {i: word for i, word in enumerate(word_vocab)}

tag2idx = {tag: i for i, tag in enumerate(tag_vocab)}
idx2tag = {i: tag for i, tag in enumerate(tag_vocab)}

# Prepare data for Seq2Seq model
max_len = max(len(s.split()) for s in input_texts)
encoder_input_data = [[word2idx[word] for word in sent.split()] for sent in input_texts]
encoder_input_data = pad_sequences(encoder_input_data, maxlen=max_len, padding='post', value=word2idx["<PAD>"])

decoder_input_data = [[tag2idx["<START>"]] + [tag2idx[tag] for tag in tags] for tags in target_texts]
decoder_output_data = [[tag2idx[tag] for tag in tags] + [tag2idx["<END>"]] for tags in target_texts]
decoder_input_data = pad_sequences(decoder_input_data, maxlen=max_len + 1, padding='post', value=tag2idx["<PAD>"])
decoder_output_data = pad_sequences(decoder_output_data, maxlen=max_len + 1, padding='post', value=tag2idx["<PAD>"])

decoder_output_data_one_hot = np.zeros(
    (len(input_texts), max_len + 1, len(tag_vocab)),
    dtype='float32'
)

for i, tags in enumerate(decoder_output_data):
    for t, tag_idx in enumerate(tags):
        decoder_output_data_one_hot[i, t, tag_idx] = 1.0

# Model architecture
latent_dim = 64
num_encoder_tokens = len(word_vocab)
num_decoder_tokens = len(tag_vocab)

# Encoder
encoder_inputs = Input(shape=(None,))
# CORRECTED: Use an Embedding layer for the encoder input.
encoder_embedding = Embedding(num_encoder_tokens, latent_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
# CORRECTED: Use an Embedding layer for the decoder input.
decoder_embedding = Embedding(num_decoder_tokens, latent_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_output_data_one_hot,
          batch_size=2, epochs=100)

# Inference (prediction) model
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_embedding_inf = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(
    decoder_embedding_inf, initial_state=decoder_states_inputs
)
decoder_states_inf = [state_h_inf, state_c_inf]
decoder_outputs_inf = decoder_dense(decoder_outputs_inf)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs_inf] + decoder_states_inf
)

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tag2idx["<START>"]
    
    stop_condition = False
    decoded_sentence = []
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_tag = idx2tag[sampled_token_index]
        
        if sampled_tag == "<END>" or len(decoded_sentence) > max_len:
            stop_condition = True
        else:
            decoded_sentence.append(sampled_tag)
        
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        
        states_value = [h, c]
        
    return decoded_sentence

# Predict for the input 'I love NLP'
input_sent = 'I love NLP'
input_seq = [word2idx[word] for word in input_sent.split()]
input_seq = pad_sequences([input_seq], maxlen=max_len, padding='post', value=word2idx["<PAD>"])

predicted_tags = decode_sequence(input_seq)

print(f"Input: {input_sent}")
print(f"Predicted POS tags: {predicted_tags}")



output
screenshort:(<img width="540" height="389" alt="exp 6 code3" src="https://github.com/user-attachments/assets/fda6dc8a-5a8d-4551-853a-b7f4f53648ee" />
<img width="629" height="585" alt="exp 6 code2" src="https://github.com/user-attachments/assets/74a06f34-cc7d-4f5b-a7bf-19d0a008da6e" />
<img width="619" height="588" alt="exp 6 code2 (2)" src="https://github.com/user-attachments/assets/61d93b55-5e4d-4db9-be79-67d952a27ce0" />
<img width="655" height="563" alt="exp 6 code" src="https://github.com/user-attachments/assets/77edff28-d850-4505-a79f-0ffacf526055" />
)
