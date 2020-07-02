import Model

model = Model.Model(epoch=100, input_size=(70, 70), datasetname="../dataset")
model.training()