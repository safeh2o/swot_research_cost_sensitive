import os
import numpy as np
import tensorflow as tf
np.random.seed(0)
tf.random.set_seed(0)
import keras
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10
import pandas as pd
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf

#These functions are all of the alternative cost function and weighting combinations tested - note that weighted_..._0
# means unweighted - putting them this way simplifies looping through the alternatives
def weighted_mse_0(y_true, y_pred):
    mse = K.mean((y_true - y_pred) ** 2)
    return mse

def weighted_mse_1(y_true, y_pred):
    weights = (1.1 + y_true) ** (-1)
    len = K.int_shape(y_pred)[1]
    error = K.sum(weights * (y_true - y_pred) ** 2) / K.sum(weights)
    return error

def weighted_mse_2(class1, class2, class3):
    def class_weight(y_true, y_pred):
        mask02 = K.cast(K.less_equal(y_true, class1), dtype="float32")
        mask05 = K.cast(K.less_equal(y_true, class2), dtype="float32") * K.cast(K.greater(y_true, class1),
                                                                                dtype="float32")
        mask10 = K.cast(K.less_equal(y_true, class3), dtype="float32") * K.cast(K.greater(y_true, class2),
                                                                                dtype="float32")
        maskm = K.cast(K.greater(y_true, class3), dtype="float32")
        weights = mask02 + 0.5 * mask05 + 0.25 * mask10 + 0.125 * maskm
        error = K.sum(weights * (y_true - y_pred) ** 2) / K.sum(weights)
        return error

    return class_weight

def weighted_mse_3(class1, class2, class3, freq1, freq2, freq3, freqm):
    def class_weight(y_true, y_pred):
        mask02 = K.cast(K.less_equal(y_true, class1), dtype="float32")
        mask05 = K.cast(K.less_equal(y_true, class2), dtype="float32") * K.cast(K.greater(y_true, class1),
                                                                                dtype="float32")
        mask10 = K.cast(K.less_equal(y_true, class3), dtype="float32") * K.cast(K.greater(y_true, class2),
                                                                                dtype="float32")
        maskm = K.cast(K.greater(y_true, class3), dtype="float32")
        weights = mask02 * freq1 + mask05 * freq2 + mask10 * freq3 + maskm * freqm
        error = K.sum(weights * (y_true - y_pred) ** 2) / K.sum(weights)
        return error

    return class_weight

def weighted_nse_0(y_true, y_pred):
    num = K.sum((y_pred - y_true) ** 2)
    y_bar = K.mean(y_true)
    denom = K.sum((y_true - y_bar) ** 2)
    NSE = -1. * (1. - (num / denom))
    return NSE


def weighted_nse_1(y_true, y_pred):
    weights = (1.1 + y_true) ** (-1)
    MSE = K.sum(weights * ((y_pred - y_true) ** 2)) / K.sum(weights)
    x_bar = K.sum(weights * y_true) / K.sum(weights)
    std_x = K.sum(weights * ((y_true - x_bar) ** 2)) / K.sum(weights)
    NSE = -1. * (1. - (MSE / std_x))
    return NSE

def weighted_nse_2(class1, class2, class3):
    def class_weight(y_true, y_pred):
        mask02 = K.cast(K.less_equal(y_true, class1), dtype="float32")
        mask05 = K.cast(K.less_equal(y_true, class2), dtype="float32") * K.cast(K.greater(y_true, class1),
                                                                                dtype="float32")
        mask10 = K.cast(K.less_equal(y_true, class3), dtype="float32") * K.cast(K.greater(y_true, class2),
                                                                                dtype="float32")
        maskm = K.cast(K.greater(y_true, class3), dtype="float32")
        weights = mask02 + 0.5 * mask05 + 0.25 * mask10 + 0.125 * maskm
        MSE = K.sum(weights * ((y_pred - y_true) ** 2)) / K.sum(weights)
        x_bar = K.sum(weights * y_true) / K.sum(weights)
        std_x = K.sum(weights * ((y_true - x_bar) ** 2)) / K.sum(weights)
        NSE = -1. * (1. - (MSE / std_x))
        return NSE

    return class_weight

def weighted_nse_3(class1, class2, class3, freq1, freq2, freq3, freqm):
    def class_weight(y_true, y_pred):
        mask02 = K.cast(K.less_equal(y_true, class1), dtype="float32")
        mask05 = K.cast(K.less_equal(y_true, class2), dtype="float32") * K.cast(K.greater(y_true, class1),
                                                                                dtype="float32")
        mask10 = K.cast(K.less_equal(y_true, class3), dtype="float32") * K.cast(K.greater(y_true, class2),
                                                                                dtype="float32")
        maskm = K.cast(K.greater(y_true, class3), dtype="float32")
        weights = mask02 * freq1 + mask05 * freq2 + mask10 * freq3 + maskm * freqm
        MSE = K.sum(weights * ((y_pred - y_true) ** 2)) / K.sum(weights)
        x_bar = K.sum(weights * y_true) / K.sum(weights)
        std_x = K.sum(weights * ((y_true - x_bar) ** 2)) / K.sum(weights)
        NSE = -1. * (1. - (MSE / std_x))
        return NSE

    return class_weight


def weighted_kge_0(y_true, y_pred):
    x_bar = K.mean(y_true)
    y_bar = K.mean(y_pred)
    r_num = K.sum((y_true - x_bar) * (y_pred - y_bar))
    x_denom = K.sqrt(K.sum((y_true - x_bar) ** 2))
    y_denom = K.sqrt(K.sum((y_pred - y_bar) ** 2))
    r_denom = x_denom * y_denom
    r = r_num / r_denom
    alpha = K.std(y_pred) / K.std(y_true)
    beta = y_bar / x_bar
    ED = K.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    KGE = -1. * (1. - ED)
    return KGE

def weighted_kge_1(y_true, y_pred):
    weights = (1.1 + y_true) ** (-1)
    x_bar = K.sum(weights * y_true) / K.sum(weights)
    y_bar = K.sum(weights * y_pred) / K.sum(weights)
    cov_xy = K.sum(weights * (y_true - x_bar) * (y_pred - y_bar)) / K.sum(weights)
    cov_xx = K.sum(weights * ((y_true - x_bar) ** 2)) / K.sum(weights)
    cov_yy = K.sum(weights * ((y_pred - y_bar) ** 2)) / K.sum(weights)
    r = cov_xy / K.sqrt(cov_xx * cov_yy)
    std_y = K.sqrt(K.sum(weights * ((y_pred - y_bar) ** 2)) / K.sum(weights))
    std_x = K.sqrt(K.sum(weights * ((y_true - x_bar) ** 2)) / K.sum(weights))
    alpha = std_y / std_x
    beta = y_bar / x_bar
    ED = K.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    KGE = -1. * (1. - ED)
    return KGE

def weighted_kge_2(class1, class2, class3):
    def class_weight(y_true, y_pred):
        mask02 = K.cast(K.less_equal(y_true, class1), dtype="float32")
        mask05 = K.cast(K.less_equal(y_true, class2), dtype="float32") * K.cast(K.greater(y_true, class1),
                                                                                dtype="float32")
        mask10 = K.cast(K.less_equal(y_true, class3), dtype="float32") * K.cast(K.greater(y_true, class2),
                                                                                dtype="float32")
        maskm = K.cast(K.greater(y_true, class3), dtype="float32")
        weights = mask02 + 0.5 * mask05 + 0.25 * mask10 + 0.125 * maskm
        x_bar = K.sum(weights * y_true) / K.sum(weights)
        y_bar = K.sum(weights * y_pred) / K.sum(weights)
        cov_xy = K.sum(weights * (y_true - x_bar) * (y_pred - y_bar)) / K.sum(weights)
        cov_xx = K.sum(weights * ((y_true - x_bar) ** 2)) / K.sum(weights)
        cov_yy = K.sum(weights * ((y_pred - y_bar) ** 2)) / K.sum(weights)
        r = cov_xy / K.sqrt(cov_xx * cov_yy)
        std_y = K.sqrt(K.sum(weights * ((y_pred - y_bar) ** 2)) / K.sum(weights))
        std_x = K.sqrt(K.sum(weights * ((y_true - x_bar) ** 2)) / K.sum(weights))
        alpha = std_y / std_x
        beta = y_bar / x_bar
        ED = K.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        KGE = -1. * (1. - ED)
        return KGE

    return class_weight

def weighted_kge_3(class1, class2, class3, freq1, freq2, freq3, freqm):
    def class_weight(y_true, y_pred):
        mask02 = K.cast(K.less_equal(y_true, class1), dtype="float32")
        mask05 = K.cast(K.less_equal(y_true, class2), dtype="float32") * K.cast(K.greater(y_true, class1),
                                                                                dtype="float32")
        mask10 = K.cast(K.less_equal(y_true, class3), dtype="float32") * K.cast(K.greater(y_true, class2),
                                                                                dtype="float32")
        maskm = K.cast(K.greater(y_true, class3), dtype="float32")
        weights = mask02 * freq1 + mask05 * freq2 + mask10 * freq3 + maskm * freqm
        x_bar = K.sum(weights * y_true) / K.sum(weights)
        y_bar = K.sum(weights * y_pred) / K.sum(weights)
        cov_xy = K.sum(weights * (y_true - x_bar) * (y_pred - y_bar)) / K.sum(weights)
        cov_xx = K.sum(weights * ((y_true - x_bar) ** 2)) / K.sum(weights)
        cov_yy = K.sum(weights * ((y_pred - y_bar) ** 2)) / K.sum(weights)
        r = cov_xy / K.sqrt(cov_xx * cov_yy)
        std_y = K.sqrt(K.sum(weights * ((y_pred - y_bar) ** 2)) / K.sum(weights))
        std_x = K.sqrt(K.sum(weights * ((y_true - x_bar) ** 2)) / K.sum(weights))
        alpha = std_y / std_x
        beta = y_bar / x_bar
        ED = K.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        KGE = -1. * (1. - ED)
        return KGE

    return class_weight

def weighted_ai_0(y_true, y_pred):
    num = K.sum((y_pred - y_true) ** 2)
    y_bar = K.mean(y_true)
    denom = K.sum((K.abs(y_pred - y_bar) + K.abs(y_true - y_bar)) ** 2)
    d = -1. * (1. - num / denom)
    return d

def weighted_ai_1(y_true, y_pred):
    weights = (1.1 + y_true) ** (-1)
    num = K.sum(weights * ((y_pred - y_true) ** 2)) / K.sum(weights)
    x_bar = K.sum(weights * y_true) / K.sum(weights)
    denom = K.sum((weights * K.abs(y_pred - x_bar) + weights * K.abs(y_true - x_bar)) ** 2) / K.sum(weights)
    d = -1. * (1. - num / denom)
    return d


def weighted_ai_2(class1, class2, class3):
    def class_weight(y_true, y_pred):
        mask02 = K.cast(K.less_equal(y_true, class1), dtype="float32")
        mask05 = K.cast(K.less_equal(y_true, class2), dtype="float32") * K.cast(K.greater(y_true, class1),
                                                                                dtype="float32")
        mask10 = K.cast(K.less_equal(y_true, class3), dtype="float32") * K.cast(K.greater(y_true, class2),
                                                                                dtype="float32")
        maskm = K.cast(K.greater(y_true, class3), dtype="float32")
        weights = mask02 + 0.5 * mask05 + 0.25 * mask10 + 0.125 * maskm
        num = K.sum(weights * ((y_pred - y_true) ** 2)) / K.sum(weights)
        x_bar = K.sum(weights * y_true) / K.sum(weights)
        denom = K.sum((weights * K.abs(y_pred - x_bar) + weights * K.abs(y_true - x_bar)) ** 2) / K.sum(weights)
        d = -1. * (1. - num / denom)
        return d

    return class_weight

def weighted_ai_3(class1, class2, class3, freq1, freq2, freq3, freqm):
    def class_weight(y_true, y_pred):
        mask02 = K.cast(K.less_equal(y_true, class1), dtype="float32")
        mask05 = K.cast(K.less_equal(y_true, class2), dtype="float32") * K.cast(K.greater(y_true, class1),
                                                                                dtype="float32")
        mask10 = K.cast(K.less_equal(y_true, class3), dtype="float32") * K.cast(K.greater(y_true, class2),
                                                                                dtype="float32")
        maskm = K.cast(K.greater(y_true, class3), dtype="float32")
        weights = mask02 * freq1 + mask05 * freq2 + mask10 * freq3 + maskm * freqm
        num = K.sum(weights * ((y_pred - y_true) ** 2)) / K.sum(weights)
        x_bar = K.sum(weights * y_true) / K.sum(weights)
        denom = K.sum((weights * K.abs(y_pred - x_bar) + weights * K.abs(y_true - x_bar)) ** 2) / K.sum(weights)
        d = -1. * (1. - num / denom)
        return d

    return class_weight

sites=['Bangladesh','Tanzania']
vars=['frc','swot'] #two input variable shorthands: FRC means FRC and Time ony, SWOT means FRC, Time, conductivity, and
# water temperature - results for this combination are in the SI
short_sites=['Bdesh','Tanz']
cost_functions=['mse','nse','kge','ai']
weights=[0,1,2,3]
IVS=['frc','swot']
nodes=np.array([[16,4],[16,8]]) #Number of hidden nodes selected for each site+variable combination, selection of the
# network architecture is in the SI
training_fraction=np.array([[1/3,1/3,1/3],[1/3,1/3,1/3]])
path=path #Important - put the path to the dataset here!


for s in range (0,len(sites)):
    for v in range (1,len(vars)):
        if v==0:
            df = pd.read_csv(path+"\\"+sites[s]+"_Out.csv") 
            df = pd.concat([df['se1_frc'], df['se4_lag'], df['se4_frc']], axis=1)
            df = df.dropna()
        else:
            df = pd.read_csv(path+"\\"+sites[s]+"_Out.csv") 
            df = pd.concat([df['se1_frc'], df['se4_lag'], df['se1_cond'], df['se1_wattemp'], df['se4_frc']], axis=1)
            df=df.dropna()
        X = df.drop(['se4_frc'], axis=1)
        Var_names = X.columns
        ndim = len(X.columns)
        S = X.to_numpy()
        Y = pd.Series(df['se4_frc'].values)
        Y = np.transpose(Y.to_numpy())
        Y = Y.reshape(-1, 1)
        predictors_scaler = MinMaxScaler(feature_range=(-1, 1))
        outputs_scaler = MinMaxScaler(feature_range=(-1, 1))
        predictors_scaler = predictors_scaler.fit(S)
        outputs_scaler = outputs_scaler.fit(Y)
        S_norm = predictors_scaler.transform(S)
        Y_norm = outputs_scaler.transform(Y)

        #For weighting 2 and 3, normalize the FRC group thresholds and convert into tensorflow-compatible format
        w0 = np.asarray(0.0)
        w0 = w0.reshape(1, -1)
        w0 = outputs_scaler.transform(w0)
        c0 = K.constant(w0, dtype="float32")
        w02 = np.asarray(0.2)
        w02 = w02.reshape(1, -1)
        w02 = outputs_scaler.transform(w02)
        c02 = K.constant(w02, dtype="float32")
        w05 = np.asarray(0.5)
        w05 = w05.reshape(1, -1)
        w05 = outputs_scaler.transform(w05)
        c05 = K.constant(w05, dtype="float32")
        w1 = np.asarray(1.0)
        w1 = w1.reshape(1, -1)
        w1 = outputs_scaler.transform(w1)
        c1 = K.constant(w1, dtype="float32")
        cmax = K.constant(np.max(Y_norm), dtype="float32")

        # calculate inverse frequencies for weighting 3
        inv_freq02 = 1 / ((np.sum(Y <= 0.2)) / len(Y))
        inv_freq05 = 1 / ((np.sum((Y <= 0.5) & (Y > 0.2)) / len(Y)))
        inv_freq10 = 1 / ((np.sum((Y <= 1) & (Y > 0.5)) / len(Y)))
        if np.sum(Y > 1) > 0:
            inv_freqm = 1 / ((np.sum(Y > 1)) / len(Y))
        else:
            inv_freqm = 0

        Ni = ndim
        Nh=nodes[v,s]
        S = X.to_numpy()
        predictors_scaler = predictors_scaler.fit(S)
        S_norm = predictors_scaler.transform(S)
        base_model = keras.models.Sequential()
        base_model.add(
            keras.layers.Dense(Nh, input_dim=Ni, activation="tanh", kernel_initializer="uniform",
                               bias_initializer="zeros"))
        base_model.add(keras.layers.Dense(1, kernel_initializer="uniform", bias_initializer="zeros", activation="linear"))
        optimizer = keras.optimizers.Nadam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
        base_model.compile(optimizer=optimizer, loss="mean_squared_error", loss_weights=None) #compile the base model with MSE, note this is just required for the model to save well, this will be replaced when it comes time to train the model
        path = os.getcwd()
        base_model.save(path+"\\base_network")

        early_stopping_monitor = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10,
                                                               restore_best_weights=True)

        for c in range (0,4): #Outer loop for cost function
            for d in range (0,4): #Inner loop for weighting
                cost_test='weighted_'+cost_functions[c]+'_'+str(d)

                Experiment=short_sites[s]+"_"+cost_functions[c]+"_w"+str(d)+"_"+vars[v]+"time"
                print(Experiment)
                if not os.path.exists(path + "\\" + Experiment):
                    os.makedirs(path + "\\" + Experiment)

                X_train_plot = []
                Y_train_plot = []
                Y_train_pred_plot = []

                train_RMSE = []
                train_RMSE_02 = []
                train_Recall = []

                X_test_plot = []
                Y_test_plot = []
                Y_pred_test_plot = []

                test_RMSE = []
                test_RMSE_02 = []
                test_Recall = []
                S_trainval_norm, S_test_norm, Y_trainval_norm, Y_test_norm = train_test_split(S_norm, Y_norm,
                                                                                              test_size=0.25,
                                                                                              shuffle=True, random_state=10)
                for i in range(0, 200): #Loop to train an ensemble of 200 ANNs - note the first line clears all history from the last trained model
                    tf.keras.backend.clear_session()

                    S_train_norm, S_val_norm, Y_train_norm, Y_val_norm = train_test_split(S_trainval_norm, Y_trainval_norm,
                                                                                          train_size=2/3, shuffle=True, random_state=i**2) #Re-split between training and validation in each model - improves the spread of the ensemble

                    model=keras.models.load_model(path+"\\base_network")

                    cost_dict = {
                        'weighted_mse_0': weighted_mse_0,
                        'weighted_mse_1': weighted_mse_1,
                        'weighted_mse_2': weighted_mse_2(c02, c05, c1),
                        'weighted_mse_3': weighted_mse_3(c02, c05, c1, inv_freq02, inv_freq05, inv_freq10, inv_freqm),
                        'weighted_nse_0': weighted_nse_0,
                        'weighted_nse_1': weighted_nse_1,
                        'weighted_nse_2': weighted_nse_2(c02, c05, c1),
                        'weighted_nse_3': weighted_nse_3(c02, c05, c1, inv_freq02, inv_freq05, inv_freq10, inv_freqm),
                        'weighted_kge_0': weighted_kge_0,
                        'weighted_kge_1': weighted_kge_1,
                        'weighted_kge_2': weighted_kge_2(c02, c05, c1),
                        'weighted_kge_3': weighted_kge_3(c02, c05, c1, inv_freq02, inv_freq05, inv_freq10, inv_freqm),
                        'weighted_ai_0': weighted_ai_0,
                        'weighted_ai_1': weighted_ai_1,
                        'weighted_ai_2': weighted_ai_2(c02, c05, c1),
                        'weighted_ai_3': weighted_ai_3(c02, c05, c1, inv_freq02, inv_freq05, inv_freq10, inv_freqm),
                    }

                    model.compile(optimizer=optimizer,loss=cost_dict[cost_test]) #recompile the model with the correct cost function

                    new_weights = [np.random.uniform(-0.05, 0.05, w.shape) for w in model.get_weights()]#New initial set of weights for each base learner in the ensemble
                    model.set_weights(new_weights)
                    model.fit(S_train_norm, Y_train_norm, epochs=1000, validation_data=(S_val_norm, Y_val_norm),
                              callbacks=[early_stopping_monitor], verbose=0,batch_size=len(Y_train_norm))

                    #After training, predict on the train and test data, save the training and testing data to an array for evaulation later
                    Y_pred_test_norm = model.predict(S_test_norm)
                    Y_pred_test = outputs_scaler.inverse_transform(Y_pred_test_norm)
                    Y_test = outputs_scaler.inverse_transform(Y_test_norm)
                    X_test = predictors_scaler.inverse_transform(S_test_norm)

                    X_test_plot = np.append(X_test_plot, X_test[:, 0])
                    Y_test_plot = np.append(Y_test_plot, Y_test)
                    Y_pred_test_plot = np.append(Y_pred_test_plot, Y_pred_test)

                    Y_pred_train_norm = model.predict(S_train_norm)
                    Y_pred_train = outputs_scaler.inverse_transform(Y_pred_train_norm)
                    Y_train = outputs_scaler.inverse_transform(Y_train_norm)
                    X_train = predictors_scaler.inverse_transform(S_train_norm)

                    X_train_plot = np.append(X_train_plot, X_train[:, 0])
                    Y_train_plot = np.append(Y_train_plot, Y_train)
                    Y_train_pred_plot = np.append(Y_train_pred_plot, Y_pred_train)

                np.save(path + "\\" + Experiment + "\\" + "X_train.npy", X_train_plot)
                np.save(path + "\\" + Experiment + "\\" + "Y_train.npy", Y_train_plot)
                np.save(path + "\\" + Experiment + "\\" + "Y_train_pred.npy", Y_train_pred_plot)
                np.save(path + "\\" + Experiment + "\\" + "X_test.npy", X_test_plot)
                np.save(path + "\\" + Experiment + "\\" + "Y_test.npy", Y_test_plot)
                np.save(path + "\\" + Experiment + "\\" + "Y_test_pred.npy", Y_pred_test_plot)
