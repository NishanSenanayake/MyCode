# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 04:35:57 2020

@author: 14198
"""


class Block(tf.keras.Model):
     def __init__(self, filters, kernel_size, repetitions, pool_size=2, strides=2):
         super(Block, self).__init__()
         self.filters = filters
         self.kernel_size = kernel_size
         self.repetitions = repetitions
        
         # Define a conv2D_0, conv2D_1, etc based on the number of repetitions
         for i in range(self.repetitions):
            
#             # Define a Conv2D layer, specifying filters, kernel_size, activation and padding.
             vars(self)[f'conv2D_{i}'] = tf.keras.layers.Conv2D(filters, kernel_size, padding='same',activation='relu')
        
#         # Define the max pool layer that will be added after the Conv2D blocks
         self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2,2))
  
     def call(self, inputs):
#         # access the class's conv2D_0 layer
         conv2D_0 =self.conv2D_0
        
#         # Connect the conv2D_0 layer to inputs
         x =conv2D_0(inputs)

#         # for the remaining conv2D_i layers from 1 to `repetitions` they will be connected to the previous layer
         for i in range(1,self.repetitions):
#             # access conv2D_i by formatting the integer `i`. (hint: check how these were saved using `vars()` earlier)
             conv2D_i = vars(self)[f'conv2D_{i}']
            
#             # Use the conv2D_i and connect it to the previous layer
             x = conv2D_i(x)

#         # Finally, add the max_pool layer
         max_pool = self.max_pool(x)
        
         return 


class MyVGG(tf.keras.Model):

     def __init__(self, num_classes):
            super(MyVGG, self).__init__()
            

#         # Creating blocks of VGG with the following 
#         # (filters, kernel_size, repetitions) configurations
            self.block_a = Block(64,3,2)
    
            self.block_b = Block(128,3,2)
            self.block_c = Block(256,3,3)
            self.block_d = Block(512,3,3)
            self.block_e = Block(512,3,3)

#         # Classification head
#         # Define a Flatten layer
            self.flatten = tf.keras.layers.Flatten()
#         # Create a Dense layer with 256 units and ReLU as the activation function
            self.fc = tf.keras.layers.Dense(256, activation='relu')
#         # Finally add the softmax classifier using a Dense layer
            self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

     def call(self, inputs):
         # Chain all the layers one after the other
         x = self.block_a(inputs)
         x = self.block_b(x)
         x = self.block_c(x)
         x = self.block_d(x)
         x = self.block_e(x)
         x = self.flatten(x)
         x = self.fc(x)
         x = self.classifier(x)
         return x





dataset = tfds.load('cats_vs_dogs', split=tfds.Split.TRAIN, data_dir='data/')

# Initialize VGG with the number of classes 
vgg = MyVGG(num_classes=2)

# Compile with losses and metrics
vgg.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define preprocessing function
def preprocess(features):
    # Resize and normalize
    image = tf.image.resize(features['image'], (224, 224))
    return tf.cast(image, tf.float32) / 255., features['label']

# Apply transformations to dataset
dataset = dataset.map(preprocess).batch(32)

# Train the custom VGG model
vgg.fit(dataset, epochs=10)