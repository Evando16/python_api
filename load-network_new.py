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
typeTrain = 'TREINAMENTO APOIO MEDIO'

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
wb = load_workbook(filename='dados.xlsx', read_only=True)
ws = wb[typeTrain]


# read xlxs
data = []
index = 0
for row in ws.rows:
    data.append([])

    for cell in row:
        data[index].append(cell.value)
    
    index += 1

# output
output = []
for i in range(0, 21):
    output.append([])
    output[i].append(data[0][i])

# input
input = []
for i in range(1, len(data)):
    for j in range(0, len(data[i])):
        if i == 1:
            input.append([])
            input[j].append(1)

        input[j].append(data[i][j])

# Aqui se e feito input[0] ocorre erro ao fazer session run
x_train = input
y_train = output #[[0], [0], [1]]


sess=tf.Session()    

#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('my_model')
saver.restore(sess,tf.train.latest_checkpoint('./'))

# Access saved Variables directly
# print(sess.run('bias:0'))
# This will print 2, which is the value of bias that we saved


# Now, let's access and create placeholders variables and
# create feed-dict to feed new data

graph = tf.get_default_graph()
# w = graph.get_tensor_by_name("W:0")
x = graph.get_tensor_by_name("X:0")
y = graph.get_tensor_by_name("Y:0")

#feed_dict = {w1:13.0, w2:17.0}

#Now, access the op that you want to run. 
# op_to_restore = graph.get_tensor_by_name("smoke:0")

W = sess.run('W:0')
op_to_restore = tf.sigmoid(tf.matmul(x, W), name='smoke')
print(sess.run(op_to_restore, {x:x_train, y:y_train}))
#This will print 60 which is calculated 

#---
# resp = jsonify((sess.run(a, {x:x_train, y:y_train}).tolist()))
# resp.status_code = 200
# return json.dumps(sess.run(op_to_restore, {x1:y_train, x1:x_train}).tolist())
#return 'ok'
