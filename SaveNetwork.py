import os
from openpyxl import load_workbook
import matplotlib.pyplot as plt
import tensorflow as tf
from SmoveHelper import TrainRule
from SmoveHelper import Rules

# Rede
rule = Rules.loadRule()

def isNaN(num):
    return num != num

# read input data
wb = load_workbook(filename='dados.xlsx', read_only=True)
ws = wb[rule.trainingType]

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
            # input[j].append(1)
        input[j].append(data[i][j])

#print('input', input)

w = tf.Variable(tf.zeros([len(input[0]), 1]), tf.float32, name='w')
x = tf.placeholder(tf.float32, [None, len(input[0])], name='X')
y = tf.placeholder(tf.float32, [None, 1], name='Y')

pred = tf.nn.sigmoid(tf.matmul(x, w), name = 'smoke')
loss = tf.reduce_mean(-tf.log(pred * y + (1 - pred) * (1 - y)))
train_step = tf.train.GradientDescentOptimizer(rule.learningRate).minimize(loss)

x_train = input #[1, 34.62, 78.02]
y_train = output #[[0], [0], [1]]

print("\n\nStarting process\n")
print(rule.trainingType)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

lastError = 0

for epoch in range(rule.rangeTrain): 
    result = sess.run([train_step, loss, w], {x: x_train, y: y_train})  

    error = result[1]

    if(isNaN(error)):
        print('return NaN')
        break

    if(error < rule.minError):
        print('finish OK')
        break

    if epoch % 5000 == 0:
        print(("%0.1f" % (rule.minError * 100 / error)))

    lastError = error

print("\n\nlastError:")
print(lastError)

# Prepara para salvar a rede
saver = tf.train.Saver()
saver.save(sess, rule.completePath, global_step=1000) 
        
print("\n\n*** Results ***\n\n")
#print('Final result:\nloss = ', result[1], '\nw = ', result[2])
print(sess.run(pred, {x:x_train, y:y_train}))    