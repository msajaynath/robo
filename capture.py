
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from  flask import Flask
import threading
from threading import Thread
app = Flask(__name__)
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import logging
import random
import time

import numpy as np
import tensorflow as tf
import os





def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph
model_file = "retrained_graph.pb"
label_file = "retrained_labels.txt"
input_height = 299
input_width = 299
input_mean = 0
input_std = 255
input_layer = "Mul"
output_layer = "final_result"

    # Load TensorFlow Graph from disk
graph = load_graph(model_file)

    # Grab the Input/Output operations
input_name = "import/" + input_layer
output_name = "import/" + output_layer
input_operation = graph.get_operation_by_name(input_name);
output_operation = graph.get_operation_by_name(output_name);

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label
def Analyse():
  camera = cv2.VideoCapture(0)
  i = 0
  stop = 0
  isOccupied = 0

  while (stop == 0):
   
    return_value, image = camera.read()
    cv2.imwrite('images/'+'test'+'.png', image)
    i=i+1
    t = read_tensor_from_image_file('images/'+'test'+'.png',
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)
        
    with tf.Session(graph=graph) as sess:
                start = time.time()
                results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
                end=time.time()
                results = np.squeeze(results)

                top_k = results.argsort()[-5:][::-1]
                labels = load_labels(label_file)

                print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))

                for i in top_k:
                    print(labels[i], results[i])
                noProbability = 0.00
                
                if labels[0] == 'no':
                  noProbability = results[0]
                else:
                  noProbability = results[1]
                if noProbability >= .90 and isOccupied == 1:
                  print ('Table empty detected!!!')
                  stop = 1
                elif noProbability < .80:
                  isOccupied = 1
                print (isOccupied)

                
                
  del(camera)

if __name__ == '__main__':
    # TensorFlow configuration/initialization
    model_file = "retrained_graph.pb"
    label_file = "retrained_labels.txt"
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer = "Mul"
    output_layer = "final_result"

    # Load TensorFlow Graph from disk
    graph = load_graph(model_file)

    # Grab the Input/Output operations
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);
    print ('test')
    Thread(target = Analyse).start()
    ##Thread(target = MainSocket).start()
hostName = "localhost"
hostPort = 9000

class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes("<html><head><title>Title goes here.</title></head>", "utf-8"))
        self.wfile.write(bytes("<body><p>This is a test.</p>", "utf-8"))
        self.wfile.write(bytes("<p>You accessed path: %s</p>" % self.path, "utf-8"))
        self.wfile.write(bytes("</body></html>", "utf-8"))

myServer = HTTPServer((hostName, hostPort), MyServer)
print(time.asctime(), "Server Starts - %s:%s" % (hostName, hostPort))

try:
    myServer.serve_forever()

except KeyboardInterrupt:
    pass

myServer.server_close()
print(time.asctime(), "Server Stops - %s:%s" % (hostName, hostPort))