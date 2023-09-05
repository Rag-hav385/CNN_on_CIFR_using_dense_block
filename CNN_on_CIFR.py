#!/usr/bin/env python
# coding: utf-8

# ### CNN on CIFR 

# In[1]:


from tensorflow.keras import models, layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Activation, Flatten
from tensorflow.keras.optimizers import Adam

import tensorflow as tf


# In[2]:


#Setting dropout rate to 0
batch_size = 150
num_classes = 10
epochs = 10
l = 40
num_filter = 12
compression = 0.5
dropout_rate = 0.0


# In[3]:


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
img_height, img_width, channel = X_train.shape[1],X_train.shape[2],X_train.shape[3]

# convert to one hot encoing 
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes) 

X_train.shape , X_test.shape


# In[4]:


# Dense Block
def denseblock(input, num_filter = 12, dropout_rate = 0.2):
    global compression
    temp = input
    for _ in range(l): 
        BatchNorm = layers.BatchNormalization()(temp)
        relu = layers.Activation('relu')(BatchNorm)
        Conv2D_3_3 = layers.Conv2D(int(num_filter*compression), (3,3), use_bias=False ,padding='same')(relu)
        if dropout_rate>0:
            Conv2D_3_3 = layers.Dropout(dropout_rate)(Conv2D_3_3)
        concat = layers.Concatenate(axis=-1)([temp,Conv2D_3_3])
        
        temp = concat
        
    return temp

## transition Blosck
def transition(input, num_filter = 12, dropout_rate = 0.2):
    global compression
    BatchNorm = layers.BatchNormalization()(input)
    relu = layers.Activation('relu')(BatchNorm)
    Conv2D_BottleNeck = layers.Conv2D(int(num_filter*compression), (1,1), use_bias=False ,padding='same')(relu)
    if dropout_rate>0:
         Conv2D_BottleNeck = layers.Dropout(dropout_rate)(Conv2D_BottleNeck)
    avg = layers.AveragePooling2D(pool_size=(2,2))(Conv2D_BottleNeck)
    return avg

#output layer
def output_layer(input):
    global compression
    BatchNorm = layers.BatchNormalization()(input)
    relu = layers.Activation('relu')(BatchNorm)
    AvgPooling = layers.AveragePooling2D(pool_size=(2,2))(relu)
    flat = layers.Flatten()(AvgPooling)
    output = layers.Dense(num_classes, activation='softmax')(flat)
    return output


# In[5]:


num_filter = 37
dropout_rate = 0.0
l = 12
input = layers.Input(shape=(img_height, img_width, channel,))
First_Conv2D = layers.Conv2D(num_filter, (1,1), use_bias=False ,padding='same')(input)

First_Block = denseblock(First_Conv2D, num_filter, dropout_rate)
First_Transition = transition(First_Block, num_filter, dropout_rate)

Second_Block = denseblock(First_Transition, num_filter, dropout_rate)
Second_Transition = transition(Second_Block, num_filter, dropout_rate)

Third_Block = denseblock(Second_Transition, num_filter, dropout_rate)
Third_Transition = transition(Third_Block, num_filter, dropout_rate)

Last_Block = denseblock(Third_Transition,  num_filter, dropout_rate)
output = output_layer(Last_Block)


# In[6]:


model = Model(inputs=[input], outputs=[output])
model.summary()


# In[7]:


print(len(model.layers))


# In[8]:


model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])


# In[9]:


#Callbacks 
import datetime
class stop_at_acc(tf.keras.callbacks.Callback):

    def __init__(self , max):
      self.max_acc = max


    def on_epoch_end(self, epoch, logs={}):
      acc = logs.get('val_accuracy')
      if(logs.get('val_accuracy') > 0.90):   
        print("\nReached %2.2f%% accuracy, so stopping training!!" %(acc*100))   
        self.model.stop_training = True
      else:
          self.max_acc = max(self.max_acc ,acc)
          print("Max accuracy till now is ",self.max_acc)

stop_at_90 = stop_at_acc(0)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.1,patience=0, min_lr=0.001)
file_path = "model_save/weights-{epoch:02d}-{val_accuracy:.4f}.hdf5"
model_check = tf.keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_accuracy', verbose=1,save_best_only=True, mode='auto')
log_dir="logs1\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1, write_graph=True,write_grads=True)

callback_list = [stop_at_90 , tensorboard_callback]


# In[10]:


#Image data generator

datagen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.4 ,height_shift_range=0.4 , horizontal_flip = True)

iterator = datagen.flow(X_train, y_train , batch_size=batch_size)


# In[11]:


model.fit(iterator, steps_per_epoch=334 ,epochs=299,verbose=1,validation_data=(X_test, y_test) , callbacks = callback_list)


# In[12]:


print("Evaluate on test data")
results = model.evaluate(X_test, y_test, batch_size=batch_size)
print("test loss, test acc:", results)


# In[13]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[14]:


get_ipython().run_line_magic('tensorboard', '--logdir  .')

