# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 20:57:29 2018

@author: Brian
"""

import tensorflow as tf

filename_queue = tf.train.string_input_producer(["../Data/train.csv"])

reader = tf.TextLineReader()
_, csv_row  = reader.read(filename_queue)

record_defaults = [[0], [0], [0], [""], [""], [0], [0], [0], [""], [0.0]]
PassengerID, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare = tf.decode_csv(value, record_defaults=record_defaults)


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    while True:
        try:
            name_data =sess.run([Name])
            print(name_data)
        except tf.errors.OutOfRangeError:
            break
