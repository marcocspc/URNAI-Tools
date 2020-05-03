class ModelBuilder():

    LAYER_INPUT = 'input'
    LAYER_OUTPUT = 'output'
    LAYER_FULLY_CONNECTED = 'fullyconn'
    LAYER_CONVOLUTIONAL = 'conv'

    DEFAULT_BUILD_MODEL = [ 
        {
            'type' : LAYER_INPUT,
            'nodes' : 50,
            'shape' : [None, 10],
        },
        {
            'type' : LAYER_FULLY_CONNECTED,
            'nodes' : 50,
            'name' : 'fc1',
        },
        {
            'type' : LAYER_FULLY_CONNECTED,
            'nodes' : 50,
            'name' : 'fc2',
        },
        {
            'type' : LAYER_OUTPUT,
            'length' : 50, 
        },
    ]

    DEFAULT_BUILD_CONV_MODEL = [ 
        {
            'type' : LAYER_INPUT,
            'nodes' : 50,
            'shape' : [None, 10],
        },
        {
            'type' : LAYER_CONVOLUTIONAL,
            'filters': 1,
            'filter_shape' : (3, 3),
            'padding' : 'same',
            'name' : 'conv1',
            'max_pooling_pool_size_shape' : (2, 2),
        },
        {
            'type' : LAYER_FULLY_CONNECTED,
            'nodes' : 50,
            'name' : 'fc1',
        },
        {
            'type' : LAYER_FULLY_CONNECTED,
            'nodes' : 50,
            'name' : 'fc2',
        },
        {
            'type' : LAYER_OUTPUT,
            'length' : 50, 
        },
    ]

    def __init__(self):
        self.layers = []

    def add_input_layer(self, size, nodes = 50, custom_shape = None):
        shape = None
        if custom_shape == None: 
            shape = [None, size]
        else: 
            shape = custom_shape

        if type(shape) == list:
            self.layers.append({
                'type' : ModelBuilder.LAYER_INPUT,
                'nodes' : nodes,
                'shape' : shape 
                })
        else:
            raise TypeError("Input layer shape should be a list with its dimensions in it.")

    def add_convolutional_layer(self, filters = 1, filter_shape = (3, 3), padding = 'same', name = 'default', input_shape = None, max_pooling_pool_size_shape = (2, 2)):
        if name == "default":
            cont = 0
            for layer in self.layers:
                if "name" in layer:
                    if "default" in layer['name']:
                        cont += 1

            name = "default" + str(cont)

        if type(filters) == int:
            if type(filter_shape) == tuple:
                if padding == 'same' or padding == 'valid':
                    self.layers.append({
                    'type' : ModelBuilder.LAYER_CONVOLUTIONAL,
                    'filters': filters,
                    'filter_shape' : filter_shape,
                    'padding' : padding,
                    'name' : name,
                    'input_shape' : input_shape,
                    'max_pooling_pool_size_shape' : max_pooling_pool_size_shape,
                    }
                    )
                else:
                    raise TypeError("Convolutional layer's padding should be 'same' or 'valid'.")
            else: 
                raise TypeError("Convolutional layer's filter_shape should be a tuple.")
        else:
            raise TypeError("Convolutional layer's filters should be an integer.")


    def add_fullyconn_layer(self, nodes, name = "default"):
        if name == "default":
            cont = 0
            for layer in self.layers:
                if "name" in layer:
                    if "default" in layer['name']:
                        cont += 1

            name = "default" + str(cont)

        if type(nodes) == int:
            self.layers.append({
                'type' : ModelBuilder.LAYER_FULLY_CONNECTED,
                'nodes' : nodes,
                'name' : name,
                })
        else:
            raise TypeError("Fully connected layer's number of nodes should be an integer.")

    def add_output_layer(self, length):
        if type(length) == int:
            self.layers.append({
                'type' : ModelBuilder.LAYER_OUTPUT,
                'length' : length,
                })
        else:
            raise TypeError("Output layer's length should be an integer.")

    def get_model_layout(self):
        return self.layers
