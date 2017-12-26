# from flask import Flask, request, Response, jsonify
# from flask_json import FlaskJSON
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from openpyxl import load_workbook
import tensorflow as tf
# import json

# app = Flask(__name__)
# FlaskJSON(app)

# @app.route('/', methods=['GET'])
# def index():
#     return 'Hello, Smoking!'

# @app.route('/smoke/network/train', methods=['POST'])
# def train():
rangeTrain = 100000    
typeTrain = 'TREINAMENTO APOIO MEDIO'

# if 'range' in request.args:
#     rangeTrain = int(request.args['range'])

# print('Range: '+ str(rangeTrain))

# if 'typeTrain' in request.args:
#     typeTrain = request.args['typeTrain']
# else:
#     return json.dumps('Informe o tipo do treinamento')  

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

#print(data[0])

# output
output = []
for i in range(0, len(data[0])):
    output.append([])
    output[i].append(data[0][i])

# ###print('output', output)
# ###print('\n')

# input
input = []
for i in range(1, len(data)):
    for j in range(0, len(data[i])):
        if i == 1:
            input.append([])
            input[j].append(1)

        input[j].append(data[i][j])

#print('input', input)

W = tf.Variable(tf.zeros([len(input[0]), 1]), tf.float32, name='W')
x = tf.placeholder(tf.float32, [None, len(input[0])], name='X')
y = tf.placeholder(tf.float32, [None, 1], name='Y')

# ###print('X: ', x)
#print('W: ', W)

a = tf.sigmoid(tf.matmul(x, W), name='smoke')

#print(a)

loss = tf.reduce_mean(- (y * tf.log(a) + (1 - y) * tf.log(1 - a)))
train_step = tf.train.GradientDescentOptimizer(1e-5).minimize(loss)

x_train = input #[1, 34.62, 78.02]
y_train = output #[[0], [0], [1]]
# print(input[0])

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()



for epoch in range(rangeTrain):
    result = sess.run([train_step, loss, W], {x: x_train, y: y_train})

saver = tf.train.Saver()
saver.export_meta_graph('my_model')

#print('Final result:\nloss = ', result[1], '\nW = ', result[2])

print(sess.run(a, {x:x_train, y:y_train}))    

# tf.train.write_graph(sess.graph_def, '.', 'smovetf.pbtxt') 
# saver.save(sess, 'smovetf.ckpt')

# return json.dumps(sess.run(a, {x:x_train, y:y_train}).tolist())