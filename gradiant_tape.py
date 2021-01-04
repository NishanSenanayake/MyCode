# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 17:36:17 2021

@author: Nishan Senanayake
"""


import tensorflow as tf




# Define a 2x2 array of 1's
x = tf.ones((2,2))

with tf.GradientTape() as t:
    # Record the actions performed on tensor x with `watch`
    t.watch(x) 

    # Define y as the sum of the elements in x
    y =  tf.reduce_sum(x)

    # Let z be the square of y
    z = tf.square(y) 

# Get the derivative of z wrt the original input tensor x
dz_dx = t.gradient(z, x)

# Print our result
print(dz_dx)



x = tf.constant(3.0)

# Notice that persistent is False by default
with tf.GradientTape() as t:
    t.watch(x)
    
    # y = x^2
    y = x * x
    
    # z = y^2
    z = y * y

# Compute dz/dx. 4 * x^3 at x = 3 --> 108.0
dz_dx = t.gradient(z, x)
print(dz_dx)

y =(3*n*n*n) -(2*n*n) + n

        # Obtain the sum of the elements in variable y
z = tf.reduce_sum(y)



#If you want to compute multiple gradients, note that by default, 
#GradientTape is not persistent (persistent=False). 
#This means that the GradientTape will expire after you use it to calculate a gradient.

x = tf.constant(3.0)

# Set persistent=True so that you can reuse the tape
with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    
    # y = x^2
    y = x * x
    
    # z = y^2
    z = y * y

# Compute dz/dx. 4 * x^3 at x = 3 --> 108.0
dz_dx = t.gradient(z, x)
print(dz_dx)

dy_dx = t.gradient(y, x)
print(dy_dx)


#Nested Gradient tapes

x = tf.Variable(1.0)

with tf.GradientTape() as tape_2:
    with tf.GradientTape() as tape_1:
        y = x * x * x
    
    # The first gradient calculation should occur at leaset
    # within the outer with block
    dy_dx = tape_1.gradient(y, x)
d2y_dx2 = tape_2.gradient(dy_dx, x)

print(dy_dx)
print(d2y_dx2)








