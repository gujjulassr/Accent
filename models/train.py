import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, History, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Conv1D, GRU, Bidirectional, Add, concatenate, Embedding, BatchNormalization, Activation, Dropout, Lambda, Multiply, LSTM, Concatenate, MaxPool1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from itertools import groupby
import random
from scipy.io import wavfile
from utils import kaldi_pad, preemphasis
from dataloader import dg_kaldi_tts
import kaldiio
import sys
from swan import MHAttn
import time

tf.profiler.experimental.start('logdir')  # Start profiler




SEED = 2
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)




def text_encoder(inputs, inp_dim):
    text_lang_emb = inputs
    conv_bank=[]
    for i in range(1,9):
        x = Conv1D(filters=128, kernel_size=i, activation='relu', padding='same')(text_lang_emb)
        x = BatchNormalization()(x)
        conv_bank.append(x)


    
  
    

    x = concatenate(conv_bank, axis=-1)

   
    x = MaxPool1D(pool_size=2, strides=1, padding='same')(x) 

    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv1D(filters=inp_dim, kernel_size=3, activation=None, padding='same')(x)
    x = BatchNormalization()(x)
   
    x = x + inputs

    x = Bidirectional(GRU(64, return_sequences=True))(x)
    
    return x


def duration_predictor(inputs,n_layers=4, kernel_size=3, dropout_rate=0.1):
    x=inputs
    for _ in range(n_layers):
        x = Conv1D(filters=128, kernel_size=kernel_size, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)  
        x = Dropout(dropout_rate)(x)


    x = Dense(96,activation='relu')(x) 
    est_dur=Dense(64, activation='relu')(x)
    est_dur=Dense(64, activation='relu')(est_dur)
    est_dur=Dense(1, activation='relu')(est_dur)
    return est_dur



def new_acoustic_decoder(inputs, n_blocks=1, n_heads=4, head_size=64, context=10, inter_dim=128, out_dim=128):
    x = Dense(inter_dim, activation='relu')(inputs)
    for i in range(n_blocks):
        cx =  MHAttn(n_heads, head_size, context)(x)
        x = BatchNormalization()(x+cx)
        xe = Dense(inter_dim, activation='relu')(x)
        xe = Dense(inter_dim, activation='relu')(xe)
        x=tf.keras.layers.BatchNormalization()(xe+x)
        x = BatchNormalization()(x+cx)


        xe = Dense(inter_dim, activation='relu')(x)
        xe = Dense(inter_dim, activation='relu')(xe)
        x=tf.keras.layers.BatchNormalization()(xe+x)
        x = BatchNormalization()(x+cx)


        xe = Dense(inter_dim, activation='relu')(x)
        xe = Dense(inter_dim, activation='relu')(xe)
        x=tf.keras.layers.BatchNormalization()(xe+x)
        x = BatchNormalization()(x+cx)


        xe = Dense(inter_dim, activation='relu')(x)
        xe = Dense(inter_dim, activation='relu')(xe)
        x=tf.keras.layers.BatchNormalization()(xe+x)
        x = BatchNormalization()(x+cx)


        xe = Dense(inter_dim, activation='relu')(x)
        xe = Dense(inter_dim, activation='relu')(xe)
        x=tf.keras.layers.BatchNormalization()(xe+x)                                    
    
    x = Dense(out_dim, activation='relu')(x)
    return x



def acoustic_decoder(inputs):
   
    conv_bank=[]
    for i in range(1,11):
        x = Conv1D(filters=128, kernel_size=i+1, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        conv_bank.append(x)

    x = concatenate(conv_bank, axis=-1)
    x = MaxPool1D(pool_size=2, strides=1, padding='same')(x) 
  
    
    x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Bidirectional(GRU(64, return_sequences=True))(x)
    return x




folder_1 = 'speaker_audio_paths.txt'


dg_train = dg_kaldi_tts(folder_1,batch_size=48, shuffle=True)



dg_val = dg_kaldi_tts(folder_1, batch_size=48, shuffle=True)


X, Y = dg_train.__getitem__(0)

output_signature = (
    (
        tf.TensorSpec(shape=(None, None, 768), dtype=tf.float32),  # fea_lis (features list)
        tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),    # phn_mask (phoneme mask)
        tf.TensorSpec(shape=(None, None, 2), dtype=tf.float32),    # phn_repeats (phoneme repeats)
        tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),    # len_mask (length mask)
        tf.TensorSpec(shape=(None,256), dtype=tf.float32),
    ),
    (
        tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),    # phn_freq (phoneme frequency)
        tf.TensorSpec(shape=(None, None, 80), dtype=tf.float32)    # mel_lis (mel spectrogram)
    )
)


# Convert your generator into a TensorFlow dataset
dg_train_tf = tf.data.Dataset.from_generator(
    lambda: dg_train,  # Your data generator
    output_signature=output_signature
).repeat().prefetch(tf.data.experimental.AUTOTUNE)

dg_val_tf = tf.data.Dataset.from_generator(
    lambda: dg_val,  # Validation generator
    output_signature=output_signature
).repeat().prefetch(tf.data.experimental.AUTOTUNE)

# .prefetch(tf.data.experimental.AUTOTUNE)







phn_lab=tf.keras.Input(shape=(None, 768), dtype=tf.float32)
phn_mask=tf.keras.Input(shape=(None,1), dtype=tf.float32)
phn_repeats=tf.keras.Input(shape=(None,2), dtype=tf.int32)

len_mask=tf.keras.Input(shape=(None,1), dtype=tf.float32)
spkr_lab_enc=tf.keras.Input(shape=(None,256), dtype=tf.int32)


                       
encoder_output = text_encoder(phn_lab, inp_dim=768)







text_spkr_emb = Concatenate(axis=-1)([encoder_output, spkr_lab_enc])

# #Duration Estimation
# est_dur=Dense(64, activation='relu')(text_spkr_emb)
# est_dur=Dense(64, activation='relu')(est_dur)
# est_dur=Dense(1, activation='relu')(est_dur)
# est_dur = Multiply(name='dur')([est_dur, phn_mask])

est_dur=duration_predictor(text_spkr_emb)
est_dur = Multiply(name='dur')([est_dur, phn_mask])











x = Lambda(lambda x: tf.gather_nd(x[0],x[1]),output_shape=(None,384))([text_spkr_emb, phn_repeats])










upsampled_enc = Multiply()([x, len_mask])












x = acoustic_decoder(upsampled_enc)



x = new_acoustic_decoder(x, n_blocks=3, n_heads=4, head_size=64, context=10, inter_dim=128, out_dim=128)

est_mel = Dense(80, activation='relu')(x)
mel_gate = Dense(80, activation='sigmoid')(x)
est_mel = Multiply()([est_mel, mel_gate])
est_mel = Multiply(name='mel')([est_mel, len_mask])





model = tf.keras.Model(inputs=[phn_lab, phn_mask, phn_repeats , len_mask, spkr_lab_enc], 
        outputs=[est_dur, est_mel])
model.summary()
#model.load_weights('dur_model.h5')
lr= 1e-4
#lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#        lr, decay_steps=303*5, decay_rate=0.95, staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
start_time = time.time()

model.compile(optimizer=optimizer, loss={'dur': 'mae','mel': 'mae',}, 
                                   loss_weights={'dur':30,'mel':10})

print(f"Model compilation took: {time.time() - start_time} seconds")

model_check = ModelCheckpoint('weights_mel_tap_text_enc/weights-{epoch:04d}.keras', monitor='val_loss')  #, save_best_only=True
early_stop = EarlyStopping(monitor='val_loss', patience=100)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, min_lr=1e-8, verbose=True, min_delta=0)
EPOCHS=350



history=model.fit(dg_train, 
        validation_data=dg_val, 
        epochs=EPOCHS, verbose=1, 
        callbacks=[model_check, early_stop, reduce_lr],
        )



tf.profiler.experimental.stop()  # Stop profiler