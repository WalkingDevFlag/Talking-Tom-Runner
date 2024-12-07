import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
from tensorflow.keras import layers

print("TensorFlow and Protobuf are working!")