#   TENSORFLOW
#   https://github.com/BinRoot/TensorFlow-Book
#   http://tensorflowbook.com/
#
#
#   MASSAS DE TREINAMENTO
#
#   TREINAMENTO- RESPOSTA A CARGA
#   TREINAMENTO APOIO TERMINAL
#   TREINAMENTO APOIO MEDIO
#   COMPARAO - XLS

TYPE_RESPOSTA_CARGA = 'TREINAMENTO- RESPOSTA A CARGA'
TYPE_APOIO_TERMINAL = 'TREINAMENTO APOIO TERMINAL'
TYPE_APOIO_MEDIO = 'TREINAMENTO APOIO MEDIO'

BASE_RESPOSTA_CARGA_PATH = './network/carga'
BASE_APOIO_TERMINAL_PATH = './network/terminal'
BASE_APOIO_MEDIO_PATH = './network/medio'

COMPLETE_RESPOSTA_CARGA_PATH = BASE_RESPOSTA_CARGA_PATH + '/carga'
COMPLETE_APOIO_TERMINAL_PATH = BASE_APOIO_TERMINAL_PATH + '/terminal'
COMPLETE_APOIO_MEDIO_PATH = BASE_APOIO_MEDIO_PATH + '/medio'

MIN_ERROR = BASE_APOIO_MEDIO_PATH + '/medio'

class Rules:

    @staticmethod
    def loadRule():
        rule = Rules.loadApoioMedio() 

        rule.rangeTrain = 200000000
        rule.learningRate = 1e-2

        return rule

    @staticmethod
    def loadRespostaCarga():
        rule = TrainRule()
        rule.trainingType = TYPE_RESPOSTA_CARGA
        rule.basePath = BASE_RESPOSTA_CARGA_PATH
        rule.completePath = COMPLETE_RESPOSTA_CARGA_PATH
        rule.offset = 20
        rule.minError = 0.000001
        return rule

    @staticmethod
    def loadApoioTerminal():
        rule = TrainRule()
        rule.trainingType = TYPE_APOIO_TERMINAL
        rule.basePath = BASE_APOIO_TERMINAL_PATH
        rule.completePath = COMPLETE_APOIO_TERMINAL_PATH
        rule.offset = 20
        rule.minError = 0.000001
        return rule

    @staticmethod
    def loadApoioMedio():
        rule = TrainRule()
        rule.trainingType = TYPE_APOIO_MEDIO
        rule.basePath = BASE_APOIO_MEDIO_PATH
        rule.completePath = COMPLETE_APOIO_MEDIO_PATH
        rule.offset = 20
        rule.minError = 0.000001
        return rule

class TrainRule:

    trainingType = ''
    basePath = ''
    completePath = ''
    offset = 0
    minError = 0
    rangeTrain = 0
    learningRate = 0