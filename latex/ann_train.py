def get_trained_ann(dataset, test_train_prop=0.25, max_epochs=50):
    tstdata, trndata = dataset.splitWithProportion(test_train_prop)
    trndata._convertToOneOfMany()
    tstdata._convertToOneOfMany()

    ann = build_simple_ann(trndata.indim, trndata.outdim)

    trainer = BackpropTrainer(ann, dataset=trndata,learningrate=0.01, momentum=0.5, verbose=True)
    trainer.trainUntilConvergence(maxEpochs=max_epochs)

    trnresult = percentError( trainer.testOnClassData(), trndata['class'] )
    tstresult = percentError( trainer.testOnClassData(dataset=tstdata ), tstdata['class'] )
    return ann, trnresult, tstresult