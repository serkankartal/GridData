import numpy as np
import  pandas as pd
from sklearn.preprocessing import  MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from flaml.ml import sklearn_metric_loss_score
import matplotlib.pyplot as plt

class Dataset:

    def get_train_test_data(self,dataset_name):
        np.random.seed(1)
        x_values =np.load("./data/" +dataset_name +".npy")
        y_temp =pd.read_csv("./data/texas_y/" +dataset_name +"_y.csv" ,delimiter=',' ,header=None)
        y_values =y_temp.iloc[: ,2]

        self.Xmax = x_values.max()
        self.Ymax = y_values.max()

        x_values = x_values/self.Xmax
        y_values = y_values/ self.Ymax
        X_train, X_test, y_train, y_test = train_test_split(x_values, y_values, test_size = 0.2, random_state = 1, shuffle=True)

        return X_train, X_test, y_train, y_test

    def evaluateResults(self,ModelName,y_test,y_pred):
        y_pred = y_pred*self.Ymax
        y_test = y_test* self.Ymax
        r2 = 1 - sklearn_metric_loss_score('r2', y_pred, y_test)
        mse = sklearn_metric_loss_score('mse', y_pred, y_test)
        mae = sklearn_metric_loss_score('mae', y_pred, y_test)

        # print("*****************************")
        print("Model Name:"+ModelName)
        print(f'r2:{r2:0.2f}')
        print(f'mse:{mse:0.2f}')
        print(f'mae:{mae:0.2f}')
        print("*****************************")

        f = open("./results/" + ModelName+"_test_scores.txt", "w")
        f.write(f'r2:{r2:0.2f}'+ '\n')
        f.write(f'mse:{mse:0.2f}' + '\n')
        f.write(f'mae:{mae:0.2f}'+ '\n')
        f.close()

        plt.plot(np.arange(len(y_test)),y_test)
        plt.plot(np.arange(len(y_pred)), y_pred)

        plt.legend(('Test Values', 'Predicted Values'), loc='upper center')
        # plt.show()
        plt.savefig("./results/" + ModelName+"_figure.png")
        plt.close()

        return r2, mse, mae