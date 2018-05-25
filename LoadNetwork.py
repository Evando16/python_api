from openpyxl import load_workbook
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from SmoveHelper import TrainRule
from SmoveHelper import Rules

# Rede
rule = Rules.loadRule()

# read input data
wb = load_workbook(filename='./dados.xlsx', read_only=True)
ws = wb['COMPARACAO - XLS']

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

saver = tf.train.import_meta_graph(rule.completePath + '-1000.meta')
saver.restore(sess,tf.train.latest_checkpoint(rule.basePath))

graph = tf.get_default_graph()
w = graph.get_tensor_by_name("w:0")
x = graph.get_tensor_by_name("X:0")
y = graph.get_tensor_by_name("Y:0")

#Now, access the op that you want to run. 
op_to_restore = graph.get_tensor_by_name("smoke:0")

results = []
count = 0
startGetCollect = 0
endGetCollect = int(rule.offset)

while(endGetCollect <= len(input)):
    x_train = []
    x_train.append(input[slice(startGetCollect, endGetCollect)])

    results.append(sess.run(op_to_restore, {x: x_train})[0][0] * 100)
    count += 1
    startGetCollect += 1
    endGetCollect += 1

results = ([0] * int(rule.offset) + results)

plt.plot(input, 'r')
plt.plot(results, 'g')
plt.ylabel(rule.trainingType)
plt.show()