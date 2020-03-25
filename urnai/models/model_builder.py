class ModelBuilder():

    LAYER_INPUT = 'input'
    LAYER_OUTPUT = 'output'
    LAYER_FULLY_CONNECTED = 'fullyconn'
    LAYER_CONVOLUTIONAL = 'conv'

    def __init__(self):
        self.layers = []

    def add_input_layer(self, shape = [None, 10]):
        if type(shape) == list:
            self.layers.append({
                'type' : ModelBuilder.LAYER_INPUT,
                'shape' : shape
                })
        else:
            raise TypeError("Input layer shape should be a list with its dimensions in it.")

    def add_fullyconn_layer(self, nodes, name):
        if type(nodes) == int:
            self.layers.append({
                'type' : ModelBuilder.LAYER_FULLY_CONNECTED,
                'nodes' : nodes,
                'name' : name,
                })
        else:
            raise TypeError("Fully connected layer's number of nodes should be an integer.")

    def add_output_layerr(self, length):
        if type(length) == int:
            self.layers.append({
                'type' : ModelBuilder.LAYER_OUTPUT,
                'length' : length,
                })
        else:
            raise TypeError("Output layer's length should be an integer.")
