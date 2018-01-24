# from flask import Flask, request, Response, jsonify
# from flask_json import FlaskJSON
from openpyxl import load_workbook
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# import json

# app = Flask(__name__)
# FlaskJSON(app)

# @app.route('/', methods=['GET'])
# def index():
#     return 'Hello, Smoking!'

# @app.route('/smoke/network/train', methods=['POST'])
# def train():
#     rangeTrain = 100000
typeTrain = 'COMPARACAO - XLS'

# if 'typeTrain' in request.args:
#     typeTrain = request.args['typeTrain']
# else:
#     resp = jsonify('Informe o tipo do treinamento')
#     resp.status_code = 400
#     return resp    

# massas
# TREINAMENTO- RESPOSTA A CARGA
# TREINAMENTO APOIO TERMINAL
# TREINAMENTO APOIO MEDIO
# COMPARAO - XLS

# read input data
wb = load_workbook(filename='./dados.xlsx', read_only=True)
ws = wb[typeTrain]

# read xlxs
data = []
index = 0
for row in ws.rows:
    data.append([])

    for cell in row:
        data[index].append(cell.value)
    
    index += 1

# input
input = []
for i in range(1, len(data)):
    input.append(data[i][1])

sess=tf.Session()    

#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('./network/terminal/terminal-1000.meta')
saver.restore(sess,tf.train.latest_checkpoint('./network/terminal'))

# Access saved Variables directly
# This will print 2, which is the value of bias that we saved

# Now, let's access and create placeholders variables and
# create feed-dict to feed new data

graph = tf.get_default_graph()
w = graph.get_tensor_by_name("w:0")
x = graph.get_tensor_by_name("X:0")
y = graph.get_tensor_by_name("Y:0")

#feed_dict = {w1:13.0, w2:17.0}

#Now, access the op that you want to run. 
op_to_restore = graph.get_tensor_by_name("smoke:0")

# print(sess.run('W:0'))
# op_to_restore = tf.sigmoid(tf.matmul(x, W), name='smoke')
# print(sess.run(op_to_restore, {x:x_train}))

results = []
count = 0
startGetCollect = 0
endGetCollect = 20

while(endGetCollect <= len(input)):
    x_train = []
    x_train.append(input[slice(startGetCollect, endGetCollect)])

    results.append(sess.run(op_to_restore, {x: x_train})[0][0])
    count += 1
    startGetCollect += 1
    endGetCollect += 1

print(results)