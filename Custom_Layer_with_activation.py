# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 20:23:24 2020

@author: 14198
"""


import tensorflow as tf
from tensorflow.keras.layers import Layer



class SimpleDense(Layer):

    # add an activation parameter
    def __init__(self, units=32, activation=None):
        super(SimpleDense, self).__init__()
        self.units = units
        
        # define the activation to get from the built-in activation layers in Keras
        self.activation = tf.keras.activations.get(activation)


    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(name="kernel",
            initial_value=w_init(shape=(input_shape[-1], self.units),
                                 dtype='float32'), trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(name="bias",
            initial_value=b_init(shape=(self.units,), dtype='float32'),
            trainable=True)
        super().build(input_shape)


    def call(self, inputs):
        
        # pass the computation to the activation layer
        return self.activation(tf.matmul(inputs, self.w) + self.b)
    
    
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    SimpleDense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)


#Assignment



class SimpleQuadratic(Layer):

    def __init__(self, units=32, activation=None):
        
        super (SimpleQuadratic,self).__init__()
        self.units=units
        self.activation=tf.keras.activations.get(activation)

    
    def build(self, input_shape):
            a_init = tf.random_normal_initializer()
            self.a=  tf.Variable(name="a",initial_value=a_init(shape=(input_shape[-1], self.units),dtype='float32'), trainable=True)
            b_init= tf.random_normal_initializer()
            self.b=  tf.Variable(name="b",initial_value=b_init(shape=(input_shape[-1], self.units),dtype='float32'), trainable=True)
            c_init= tf.zeros_initializer()
            self.c=tf.Variable(name="c",initial_value=c_init(shape=(self.units,),dtype='float32'), trainable=True)
   
    def call(self, inputs):
        x_squared=tf.math.square(inputs)
        x_squared_times_a=tf.matmul(x_squared,self.a)
        x_times_b= tf.matmul(inputs,self.b)
        x2a_plus_xb_plus_c = x_squared_times_a + x_times_b +self.c
        return self.activation(x2a_plus_xb_plus_c)



mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  SimpleQuadratic(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)




