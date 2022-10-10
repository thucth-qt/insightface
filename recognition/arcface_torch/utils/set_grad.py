def set_grad_bb(backbone, grad=True):
    '''
    Backbone must be a instance of Backbone class
    '''
    print("Setting Backbone required_grad = ", grad)
    for layer in  backbone.input_layer:
        for param in layer.parameters():
            param.requires_grad=grad
    for layer in  backbone.body:
        for param in layer.parameters():
            param.requires_grad=grad
    for layer in  backbone.output_layer:
        for param in layer.parameters():
            param.requires_grad=grad
    