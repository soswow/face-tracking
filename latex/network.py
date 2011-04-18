def build_simple_ann(indim, outdim):
    ann = FeedForwardNetwork()

    ann.addInputModule(LinearLayer(indim, name='in'))
    ann.addOutputModule(SoftmaxLayer(outdim,name='out'))
    ann.addModule(BiasUnit(name='bias'))

    ann.addConnection(FullConnection(ann['in'], ann['out']))
    ann.addConnection(FullConnection(ann['bias'], ann['out']))

    ann.sortModules()
    return ann

def build_comlex_ann(indim, outdim):
    ann = FeedForwardNetwork()

    ann.addInputModule(LinearLayer(indim, name='in'))
    ann.addModule(SigmoidLayer(10, name='hidden'))
    ann.addOutputModule(SoftmaxLayer(outdim,name='out'))
    ann.addModule(BiasUnit(name='bias'))

    ann.addConnection(FullConnection(ann['in'], ann['hidden']))
    ann.addConnection(FullConnection(ann['hidden'], ann['out']))
    ann.addConnection(FullConnection(ann['bias'], ann['out']))
    ann.addConnection(FullConnection(ann['bias'], ann['hidden']))

    ann.sortModules()
    return ann
