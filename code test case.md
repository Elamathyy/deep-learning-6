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

# Add special tokens
word_vocab.insert(0, "<PAD>")
tag_vocab.insert(0, "<PAD>")
tag_vocab.insert(1, "<START>")
tag_vocab.append("<END>")

word2idx = {word: i for i, word in enumerate(word_vocab)}
idx2word = {i: word for i, word in enumerate(word_vocab)}
tag2idx = {tag: i for i, tag in enumerate(tag_vocab)}
idx2tag = {i: tag for i, tag in enumerate(tag_vocab)}

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

latent_dim = 64
num_encoder_tokens = len(word_vocab)
num_decoder_tokens = len(tag_vocab)

# Encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(num_encoder_tokens, latent_dim, mask_zero=True)
encoder_embedded = encoder_embedding(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedded)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(num_decoder_tokens, latent_dim, mask_zero=True)
decoder_embedded = decoder_embedding(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedded, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy')

model.fit([encoder_input_data, decoder_input_data], decoder_output_data_one_hot,
          batch_size=2, epochs=100, verbose=1)

# Inference Models - reuse layers for weights sharing

# Encoder inference model
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder inference model
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Use the same embedding layer (decoder_embedding) and LSTM (decoder_lstm)
decoder_inputs_single = Input(shape=(1,))  # one token at a time
decoder_embedded_inf = decoder_embedding(decoder_inputs_single)

decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(
    decoder_embedded_inf, initial_state=decoder_states_inputs
)
decoder_states_inf = [state_h_inf, state_c_inf]
decoder_outputs_inf = decoder_dense(decoder_outputs_inf)

decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs_inf] + decoder_states_inf
)

def decode_sequence(input_seq):
    # Encode input sequence to get states
    states_value = encoder_model.predict(input_seq)

    target_seq = np.array([[tag2idx["<START>"]]])  # start token
    decoded_tags = []

    while True:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_tag = idx2tag[sampled_token_index]

        if sampled_tag == "<END>" or len(decoded_tags) > max_len:
            break

        decoded_tags.append(sampled_tag)

        # Update target_seq and states
        target_seq = np.array([[sampled_token_index]])
        states_value = [h, c]

    return decoded_tags

# Test
for test_sentence in ['I love NLP', 'He plays football']:
    test_seq = [word2idx[word] for word in test_sentence.split()]
    test_seq = pad_sequences([test_seq], maxlen=max_len, padding='post', value=word2idx["<PAD>"])
    predicted_tags = decode_sequence(test_seq)
    print(f"Input: {test_sentence}")
    print(f"Predicted POS tags: {predicted_tags}")



output
screenshot:(<img width="716" height="447" alt="ao5" src="https://github.com/user-attachments/assets/f4c07acf-c1dd-49ff-98fe-e2b3e0015388" />
<img width="815" height="535" alt="ao4" src="https://github.com/user-attachments/assets/c011e8d1-bf3f-4613-bac7-7aa6c89023a4" />
<img width="811" height="549" alt="ao3" src="https://github.com/user-attachments/assets/bcfcff7d-aacf-41bc-be0d-3b227dc3a775" />
<img width="846" height="541" alt="ao2" src="https://github.com/user-attachments/assets/8a8e23d9-4817-4380-b646-9258bf1d9ea6" />
<img width="1505" height="383" alt="ao7" src="https://github.com/user-attachments/assets/1b710cbb-5129-40cf-9b00-3ecbaa508a55" />
<img width="756" height="466" alt="ao6" src="https://github.com/user-attachments/assets/c66d9e0e-2542-4814-992c-0d01281db51e" />
)
