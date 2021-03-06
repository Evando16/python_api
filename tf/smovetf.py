import tensorflow as tf
from openpyxl import load_workbook

# massas
# TREINAMENTO- RESPOSTA A CARGA
# TREINAMENTO APOIO TERMINAL
# TREINAMENTO APOIO MEDIO
# COMPARAO - XLS

# read input data
wb = load_workbook(filename='dados.xlsx', read_only=True)
ws = wb['TREINAMENTO APOIO MEDIO']


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

print('output', output)
print('\n')

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
y = tf.placeholder(tf.float32, [None, 1])

print('X: ', x)
#print('W: ', W)

a = tf.sigmoid(tf.matmul(x, W), name='O')

#print(a)

loss = tf.reduce_mean(- (y * tf.log(a) + (1 - y) * tf.log(1 - a)))
train_step = tf.train.GradientDescentOptimizer(1e-5).minimize(loss)

x_train = input #[[1, 34.62, 78.02], [1, 33, 79], [1, 60.18, 86.30]]
y_train = output #[[0], [0], [1]]

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

saver = tf.train.Saver()

for epoch in range(100000):
    result = sess.run([train_step, loss, W], {x: x_train, y: y_train})

#print('Final result:\nloss = ', result[1], '\nW = ', result[2])
print(sess.run(a, {x:x_train, y:y_train}))

tf.train.write_graph(sess.graph_def, '.', 'smovetf.pbtxt') 
saver.save(sess, 'smovetf.ckpt')