import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import seaborn as sns
random.seed(42)
np.random.seed(42)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import copy

CUDA_NUM = '0'
device = 'cuda:'+CUDA_NUM if torch.cuda.is_available() else 'cpu'

class Automated_XGBoost:
    def __init__(self, train_y_name, train_data, valid_percentage, test_data, params):

        if params.problem_type not in ['classification', 'regression']:
            raise Exception("Problem type not written properly. Please use 'classification' or 'regression'.")

        self.train_y_name = train_y_name
        self.problem_type = params.problem_type
        self.train_data = train_data
        self.valid_percentage = valid_percentage
        self.test_data = test_data
        self.params = params

    def train(self) : 
        y = self.train_data[self.train_y_name]
        X = self.train_data.drop([self.train_y_name],axis=1)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.valid_percentage)
        XGB_start_time = time.time()
        if self.problem_type=='classification':
            XGBmodel = XGBClassifier(learning_rate = self.params.learning_rate,max_depth= self.params.max_depth,
                                  n_estimators= self.params.n_estimators,subsample= self.params.subsample,gamma= self.params.gamma)
        else:
            XGBmodel = XGBRegressor(learning_rate = self.params.learning_rate,max_depth= self.params.max_depth,
                                  n_estimators= self.params.n_estimators,subsample= self.params.subsample,gamma= self.params.gamma)
        
        XGBmodel.fit(X_train,y_train.to_numpy().ravel())
        XGB_end_time = time.time()
        XGB_run_time = XGB_end_time - XGB_start_time
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.XGBmodel = XGBmodel
        self.XGB_run_time = XGB_run_time
        

    def trainset_scores(self):
        if self.problem_type=='classification':
            XGB_train_pred = np.round(self.XGBmodel.predict(self.X_train))
            XGB_train_acc = accuracy_score(self.y_train,XGB_train_pred)
            XGB_train_prec = precision_score(self.y_train,XGB_train_pred)
            XGB_train_recall = recall_score(self.y_train,XGB_train_pred)
            XGB_train_f1 = f1_score(self.y_train,XGB_train_pred)
            return [XGB_train_acc,XGB_train_prec,XGB_train_recall,XGB_train_f1]
        else :
            XGB_train_pred = self.XGBmodel.predict(self.X_train)
            XGB_train_MAE = np.mean(np.abs((self.y_train-XGB_train_pred)))
            return [XGB_train_MAE]
    
    def validset_scores(self):
        if self.problem_type=='classification':
            XGB_val_pred = np.round(self.XGBmodel.predict(self.X_val))
            XGB_val_acc = accuracy_score(self.y_val,XGB_val_pred)
            XGB_val_prec = precision_score(self.y_val,XGB_val_pred)
            XGB_val_recall = recall_score(self.y_val,XGB_val_pred)
            XGB_val_f1 = f1_score(self.y_val,XGB_val_pred)
            return [XGB_val_acc,XGB_val_prec,XGB_val_recall,XGB_val_f1]
        else :
            XGB_val_pred = self.XGBmodel.predict(self.X_val)
            XGB_val_MAE = np.mean(np.abs((self.y_val-XGB_val_pred)))
            return [XGB_val_MAE]

    def testset_predictions(self):
        X_test = self.test_data
        if self.problem_type=='classification':
            return np.round(self.XGBmodel.predict(X_test))
        return self.XGBmodel.predict(X_test)

    def return_traintime(self):
        return self.XGB_run_time
        
    def return_feature_importance(self):
        return self.XGBmodel.feature_importances_
    
    def save_model(self,directory_name):
        self.XGBmodel.save_model(directory_name+'.bin')
    
    def return_model(self):
        return self.XGBmodel
    
    def draw_plot(self,num):
        fi = pd.Series(self.XGBmodel.feature_importances_, index=self.X_train.columns).sort_values(ascending=False)
        top_10_features = fi[:num] 
        sns.barplot(x=top_10_features, y=top_10_features.index)
        plt.show()
    
    def save_plot(self, num, route):
        fi = pd.Series(self.XGBmodel.feature_importances_, index=self.X_train.columns).sort_values(ascending=False)
        top_10_features = fi[:num] 
        sns.barplot(x=top_10_features, y=top_10_features.index)
        plt.savefig(route + '.png')
        plt.show()

        

        
def run_AE_classifier(train_data, train_y_name, valid_percentage, test_data, params):
    
    X = train_data.drop([train_y_name],axis=1)
    y = train_data[train_y_name]
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_percentage, shuffle=True)

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_valid = scaler_X.transform(X_valid)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.to_numpy().reshape(-1, 1)).reshape(-1)
    y_valid = scaler_y.transform(y_valid.to_numpy().reshape(-1, 1)).reshape(-1)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)
    X_valid = torch.tensor(X_valid, dtype=torch.float32).to(device)
    y_valid = torch.tensor(y_valid, dtype=torch.float32).reshape(-1, 1).to(device)
    
    input_ = len(X.columns)
    Regressor_model = nn.Sequential(
        nn.Linear(input_, int(input_*3)),
        nn.LeakyReLU(),
        nn.Dropout(0.2),
        nn.Linear(int(input_*3), int(input_*1.5)),
        nn.LeakyReLU(),
        nn.Dropout(0.2),
        nn.Linear(int(input_*1.5), int(input_*0.75)),
        nn.LeakyReLU(),
        nn.Dropout(0.2),
        nn.Linear(int(input_*0.75), 1)).to(device)
    
    loss_fn = nn.MSELoss().to(device)
    optimizer = optim.Adam(Regressor_model.parameters(), lr=params.lr)
    batch_start = torch.arange(0, len(X_train), params.batch_size)
    
    best_mse = np.inf   
    best_weights = None
    history = []
    patience = 5
    no_improvement = 0
    
    for epoch in range(params.n_epochs):
        Regressor_model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                X_batch = X_train[start:start+params.batch_size].to(device)
                y_batch = y_train[start:start+params.batch_size].to(device)
                y_pred = Regressor_model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bar.set_postfix(mse=float(loss))
        Regressor_model.eval()
        y_pred = Regressor_model(X_valid)
        mse = loss_fn(y_pred, y_valid)
        mse = float(mse)
        history.append(mse)

        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(Regressor_model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print("Early stopping triggered. No improvement in validation loss.")
                break
            
    Regressor_model.load_state_dict(best_weights)
    Regressor_model.eval()

    with torch.no_grad():
        y_pred_train = Regressor_model(X_train)
        mse_train_loss = loss_fn(y_pred_train, y_train)
    mse_train_loss = float(mse_train_loss)
    y_pred_train_unscaled = scaler_y.inverse_transform(y_pred_train.cpu().numpy())
    y_train_unscaled = scaler_y.inverse_transform(y_train.cpu().numpy())
    mae_train_loss = np.mean(np.abs(y_pred_train_unscaled - y_train_unscaled))
    
    with torch.no_grad():
        y_pred_valid = Regressor_model(X_valid)
        mse_valid_loss = loss_fn(y_pred_valid, y_valid)
    mse_valid_loss = float(mse_valid_loss)
    y_pred_valid_unscaled = scaler_y.inverse_transform(y_pred_valid.cpu().numpy())
    y_valid_unscaled = scaler_y.inverse_transform(y_valid.cpu().numpy())
    mae_valid_loss = np.mean(np.abs(y_pred_valid_unscaled-y_valid_unscaled))
    
    X_test = test_data.drop([train_y_name],axis=1)
    X_test = scaler_X.transform(X_test)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        final_AE_test_predictions = Regressor_model(X_test)
    AE_test_predictions = scaler_y.inverse_transform(final_AE_test_predictions.cpu().numpy())
    preds = [0 if i<0.5 else 1 for i in AE_test_predictions] 
    p,r,f1 = Precision_Recall_f1score(y_valid,preds)
    return scaler_X, scaler_y, Regressor_model , preds, [p,r,f1]


def run_AE_regressor(train_data, train_y_name, valid_percentage,test_data, params):
    
    X = train_data.drop([train_y_name],axis=1)
    y = train_data[train_y_name]
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_percentage, shuffle=True)

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_valid = scaler_X.transform(X_valid)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.to_numpy().reshape(-1, 1)).reshape(-1)
    y_valid = scaler_y.transform(y_valid.to_numpy().reshape(-1, 1)).reshape(-1)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)
    X_valid = torch.tensor(X_valid, dtype=torch.float32).to(device)
    y_valid = torch.tensor(y_valid, dtype=torch.float32).reshape(-1, 1).to(device)
    
    input_ = len(X.columns)
    Regressor_model = nn.Sequential(
        nn.Linear(input_, int(input_*3)),
        nn.LeakyReLU(),
        nn.Dropout(0.2),
        nn.Linear(int(input_*3), int(input_*1.5)),
        nn.LeakyReLU(),
        nn.Dropout(0.2),
        nn.Linear(int(input_*1.5), int(input_*0.75)),
        nn.LeakyReLU(),
        nn.Dropout(0.2),
        nn.Linear(int(input_*0.75), 1)).to(device)
    
    loss_fn = nn.MSELoss().to(device)
    optimizer = optim.Adam(Regressor_model.parameters(), lr=params.lr)
    batch_start = torch.arange(0, len(X_train), params.batch_size)
    
    best_mse = np.inf   
    best_weights = None
    history = []
    patience = 5
    no_improvement = 0
    
    for epoch in range(params.n_epochs):
        Regressor_model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                X_batch = X_train[start:start+params.batch_size].to(device)
                y_batch = y_train[start:start+params.batch_size].to(device)
                y_pred = Regressor_model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bar.set_postfix(mse=float(loss))
        Regressor_model.eval()
        y_pred = Regressor_model(X_valid)
        mse = loss_fn(y_pred, y_valid)
        mse = float(mse)
        history.append(mse)

        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(Regressor_model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print("Early stopping triggered. No improvement in validation loss.")
                break
    
    Regressor_model.load_state_dict(best_weights)
    Regressor_model.eval()

    with torch.no_grad():
        y_pred_train = Regressor_model(X_train)
        mse_train_loss = loss_fn(y_pred_train, y_train)
    mse_train_loss = float(mse_train_loss)
    y_pred_train_unscaled = scaler_y.inverse_transform(y_pred_train.cpu().numpy())
    y_train_unscaled = scaler_y.inverse_transform(y_train.cpu().numpy())
    mae_train_loss = np.mean(np.abs(y_pred_train_unscaled - y_train_unscaled))
    
    with torch.no_grad():
        y_pred_valid = Regressor_model(X_valid)
        mse_valid_loss = loss_fn(y_pred_valid, y_valid)
    mse_valid_loss = float(mse_valid_loss)
    y_pred_valid_unscaled = scaler_y.inverse_transform(y_pred_valid.cpu().numpy())
    y_valid_unscaled = scaler_y.inverse_transform(y_valid.cpu().numpy())
    mae_valid_loss = np.mean(np.abs(y_pred_valid_unscaled-y_valid_unscaled))
    
    print("Train Results : [MSE, MAE] = [{:.3f}, {:.3f}]".
                  format(mae_train_loss,mse_train_loss))
    print("Valid Results : [MSE, MAE] = [{:.3f}, {:.3f}]".
                  format(mae_valid_loss,mse_valid_loss))
    
    return scaler_X, Regressor_model , [mse_train_loss,mae_train_loss,mse_valid_loss,mae_valid_loss]

def math_round(x):
    M = x//1
    return M if x-M<0.5 else M+1

def Precision_Recall_f1score(Reals, Preds):
    TP, TN, FP, FN = 0, 0, 0, 0
    for real, pred in zip(Reals, Preds):
        if [real, pred] == [0, 0] : TN += 1
        elif [real, pred] == [0, 1] : FP += 1
        elif [real, pred] == [1, 0] : FN += 1
        else: TP += 1
    if TP+FN==0:
        P = TP/(TP+FP)
        return P,0,0
    elif TP+FP==0:
        R = TP/(TP+FN)
        return 0,R,0
    R = TP / (TP + FN)
    P = TP / (TP + FP)
    if R + P == 0:
        return R,P,0
    F1 = 2 * R * P / (R + P)
    return [P, R, F1]



