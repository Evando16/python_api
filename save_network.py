import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from openpyxl import load_workbook
import matplotlib.pyplot as plt
import tensorflow as tf

# massas
# TREINAMENTO- RESPOSTA A CARGA
# TREINAMENTO APOIO TERMINAL
# TREINAMENTO APOIO MEDIO
# COMPARAO - XLS

CARGA_PATH = './network/carga/carga'
TERMINAL_PATH = './network/terminal/terminal'
MEDIO_PATH = './network/medio/medio'

rangeTrain = 2000000
learningRate = 1e-4
minError = 0.0085
typeTrain = 'TREINAMENTO- RESPOSTA A CARGA'

def isNaN(num):
    return num != num

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
            # input[j].append(1)
        input[j].append(data[i][j])

#print('input', input)

w = tf.Variable(tf.zeros([len(input[0]), 1]), tf.float32, name='w')
x = tf.placeholder(tf.float32, [None, len(input[0])], name='X')

y = tf.placeholder(tf.float32, [None, 1], name='Y')
b = tf.Variable(tf.zeros([1]))

# ###print('X: ', x)
#print('w: ', w)

pred = tf.sigmoid(tf.matmul(x, w) + b, name='smoke')
#pred = tf.nn.softmax(tf.matmul(x, w) + b)

#print(a)

loss = tf.reduce_mean(- (y * tf.log(pred) + (1 - y) * tf.log(1 - pred)))
#loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

x_train = input #[1, 34.62, 78.02]
y_train = output #[[0], [0], [1]]
# print(input[0])

#plt.plot(input[0], input[5])
#plt.ylabel(typeTrain)
#plt.show()

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for epoch in range(rangeTrain): 
    result = sess.run([train_step, loss, w], {x: x_train, y: y_train})  

    error = result[1]

    if(isNaN(error)):
        print('return NaN')
        break

    if(error < minError):
        print('finish OK')
        break

    if epoch % 1000 == 0:
        print("%0.2f" % (minError/error))
        #print (float(epoch) / rangeTrain) * 100

saver = tf.train.Saver()

if typeTrain == 'TREINAMENTO- RESPOSTA A CARGA':
    saver.save(sess, CARGA_PATH, global_step=1000)
    print('carga')
elif typeTrain == 'TREINAMENTO APOIO TERMINAL':
    saver.save(sess, TERMINAL_PATH, global_step=1000)
    print('terminal')
elif typeTrain == 'TREINAMENTO APOIO MEDIO':
    saver.save(sess, MEDIO_PATH, global_step=1000)    
    print('medio')
        

#print('Final result:\nloss = ', result[1], '\nw = ', result[2])

print(sess.run(pred, {x:x_train, y:y_train}))    