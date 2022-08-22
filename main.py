import gridrad
import keras
import CNNModels
from GridDataset import  *

datasetName="data128_pixel_thr_30_dbz_thr_30_Rees"
model=CNNModels.Resnet34()
# model,modelName=CNNModels.LeNet5() asd
print(model.summary())
CNNModels.TrainModel(model,datasetName)
CNNModels.TestModel(datasetName,model.name)

a=3