from openpyxl import load_workbook
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from SmoveHelper import TrainRule
from SmoveHelper import Rules

# Rede
rule = Rules.loadRespostaCarga()

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

#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph(rule.completePath + '-1000.meta')
saver.restore(sess,tf.train.latest_checkpoint(rule.basePath))

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

    results.append(sess.run(op_to_restore, {x: x_train})[0][0] * 100)
    count += 1
    startGetCollect += 1
    endGetCollect += 1

#print(results)

plt.plot(input, 'r')
plt.plot(results, 'g')
plt.ylabel(rule.trainingType)
plt.show()