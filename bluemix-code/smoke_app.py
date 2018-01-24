from flask import Flask, request
import os
from openpyxl import load_workbook
import tensorflow as tf
import json
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

CARGA_PATH_NETWORK = './network/carga/carga-1000.meta'
TERMINAL_PATH_NETWORK = './network/terminal/terminal-1000.meta'
MEDIO_PATH_NETWORK ='./network/medio/medio-1000.meta'

CARGA_PATH_CHECKPOINT = './network/carga'
TERMINAL_PATH_CHECKPOINT = './network/terminal'
MEDIO_PATH_CHECKPOINT ='./network/medio/'


@app.route('/', methods=['GET'])
def index():
    return 'Hello, Smoking!'


@app.route('/api/smoke/run', methods=['POST'])
def smoke():
    collectJSON = ''
    typeTrain = ''
    result = None
    try:
        dataJSON = request.get_json()
        collectJSON = dataJSON['collect']
        typeTrain = dataJSON['type']
    except:
        return "Invalid JSON", 400

    if typeTrain == '' or typeTrain is None:
        return "Invalid type of training"
    
    if collectJSON == '' or collectJSON is None:
        return "Invalid collection data", 400

    if len(collectJSON) < 20:
        return "Insufficient collection quantity", 400

    if typeTrain.lower() == 'carga':
        print('carga----')
        result = run(collectJSON, CARGA_PATH_NETWORK, CARGA_PATH_CHECKPOINT)
    elif typeTrain.lower() == 'terminal':
        print('terminal----')
        result = run(collectJSON, TERMINAL_PATH_NETWORK, TERMINAL_PATH_CHECKPOINT)
    elif typeTrain.lower() == 'medio':
        print('medio----')
        print(MEDIO_PATH_NETWORK)
        print(MEDIO_PATH_CHECKPOINT)        
        result = run(collectJSON, MEDIO_PATH_NETWORK, MEDIO_PATH_CHECKPOINT)
    else:
        return "Type of training does not exist", 400
    
    return json.dumps(result), 200
        

def run(data, PATH_NETWORK, PATH_CHECKPOINT):
    sess = tf.Session()

    saver = tf.train.import_meta_graph(PATH_NETWORK)
    saver.restore(sess, tf.train.latest_checkpoint(PATH_CHECKPOINT))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("X:0")
    y = graph.get_tensor_by_name("Y:0")

    pred = graph.get_tensor_by_name("smoke:0")

    results = []
    count = 0
    startGetCollect = 0
    endGetCollect = 20

    while(endGetCollect <= len(data)):
        x_train = []
        x_train.append(data[slice(startGetCollect, endGetCollect)])

        results.append(sess.run(pred, {x: x_train})[0][0])
        count += 1
        startGetCollect += 1
        endGetCollect += 1

    # print(np.asarray(results).tolist())
    return np.asarray(results).tolist()

port=os.getenv('PORT', '5000')
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(port))