from flask import Flask, request
import os
from openpyxl import load_workbook
import tensorflow as tf
import json
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return 'Hello, Smoking!'


@app.route('/api/smoke/run', methods=['POST'])
def run():
    typeTrain = 'TREINAMENTO APOIO MEDIO'
    collectJSON = ''

    # Validacoes
    try:
        dataJSON = request.get_json()
        collectJSON = dataJSON['collect']
    except:
        return "Invalid JSON", 400

    if collectJSON == '' or collectJSON is None:
        return "Invalid collection data", 400

    if len(collectJSON) < 20:
        return "Insufficient collection quantity", 400

    sess = tf.Session()

    saver = tf.train.import_meta_graph('./network/smoke-1000.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./network/'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("X:0")
    y = graph.get_tensor_by_name("Y:0")

    op_to_restore = graph.get_tensor_by_name("smoke:0")

    results = []
    count = 0
    startGetCollect = 0
    endGetCollect = 20

    while(endGetCollect <= len(collectJSON)):
        x_train = []
        x_train.append(collectJSON[slice(startGetCollect, endGetCollect)])

        results.append(sess.run(op_to_restore, {x: x_train})[0])
        count += 1
        startGetCollect += 1
        endGetCollect += 1

    print(np.asarray(results))
    return json.dumps(np.asarray(results).tolist()), 200

port=os.getenv('PORT', '5000')
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(port))
