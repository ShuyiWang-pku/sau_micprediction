# encoding: utf-8
import re
import pandas as pd
import numpy as np 
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import precision_score
from sklearn.model_selection import ShuffleSplit 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import joblib

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

X_name = []
MIC = []
da = pd.read_csv("/path/to/csv",header = 1, index_col=0, low_memory=False)
print(da.head()) 

#according to 'merge-to-csv.py', remove the first line, which is 'ID'
da = da.drop('ID',axis=0)
print(da.head())

rownames = da.index

strainame_total = []
#Delete strains without drug susceptibility data
def removenull():
    f1 = open('/path/to/MICdata','r')
    data1=f1.readlines()
    f1.close()
    
    tmp = 0
    for line in data1:
        if tmp < 1:
            tmp = tmp + 1
            continue
        
        tag = 0
        x = line.split("	")

        if x[1].strip('\n') == 'NA': 
            tag = 1
            continue
        if(tag == 0):
            strain = x[0]
            strainame_total.append(strain)
        
removenull()

#isolate number left
strainame_save = []
for i in range(len(rownames)):
    tmp = 0 
    for j in range(len(strainame_total)):
        if strainame_total[j] == rownames[i]:
            tmp = 1
            break
    if(tmp == 1):
        strainame_save.append(rownames[i])
print(strainame_save)

print(len(X_name))

X_da = da.loc[da.index.intersection(X_name)]
ID = list(X_da.index.values)
X=X_da.values
print(X.shape)

ID_2 = []

def read_pheno():
    f1 = open('/path/to/MICdata','r')
    data1=f1.readlines()
    f1.close()
    
    for i in range(len(ID)):
        tmp = 0
        for line in data1:
            if tmp < 1:
                tmp = tmp + 1
                continue       
            x = line.split("	")
            strain = re.sub('"','',x[1])
        
            tmp1 = 0            
            if ID[i] == strain:
                tmp1 = 1
            if tmp1 == 0:
                continue
                
            tag = 0
            if x[1].strip('\n') == 'NA':
                tag = 1
                continue
            if(tag == 0):
                ID_2.append(strain)
                MIC.append(x[1])
    
read_pheno()

print(len(MIC))
print(ID)
#to retest the isolate name
print(ID_2)
print(MIC)

#Display the statistical results of MIC
value_cnt = {}  
for value in MIC:
    value_cnt[value] = value_cnt.get(value, 0) + 1

print(value_cnt)
print([key for key in value_cnt.keys()])
print([value for value in value_cnt.values()])

Y_MIC = []

for value in MIC:
    if value == '512':
        Y_MIC.append(13)
    if value == '256':
        Y_MIC.append(12)    
    if value == '128':
        Y_MIC.append(11)
    if value == '64':
        Y_MIC.append(10)
    if value == '32':
        Y_MIC.append(9)
    if value == '16':
        Y_MIC.append(8)
    if value == '8':
        Y_MIC.append(7)
    if value == '4':
        Y_MIC.append(6)
    if value == '1':
        Y_MIC.append(5)
    if value == '0.5':
        Y_MIC.append(4)
    if value == '0.25':
        Y_MIC.append(3)
    if value == '0.125':
        Y_MIC.append(2)
    if value == '0.064':
        Y_MIC.append(1)

#Display the statistical results of Y_test
value_cnt = {}  
for value in Y_MIC:
    value_cnt[value] = value_cnt.get(value, 0) + 1
print(value_cnt)
print([key for key in value_cnt.keys()])
print([value for value in value_cnt.values()])

Y_MIC = np.array(Y_MIC)
print(Y_MIC.shape)


def countROCAUC(y_test,y_score,drug_name,clfname,n_classes):
    #Calculate ROC curve and AUC for each category 
    y_test = np.array(y_test)
    print(y_test.shape)
    y_score = np.array(y_score)
    print(y_score.shape)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    print(fpr)
    print(tpr)
    print(roc_auc)

    return fpr,tpr,roc_auc

def trainmodel(X,Y,drug_name):   
    n_classes=len(np.unique(Y))
    print(n_classes)
    
    #LabelBinarizer - one hot
    lb=LabelBinarizer()
    Y = lb.fit_transform(Y) 
    
    X_select, X_test, y_select, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    print(X_select.shape, y_select.shape)
    print(X_test.shape, y_test.shape)
    
    global ID_2
    print(drug_name)
    
    acc = 0   
    
    sss = StratifiedShuffleSplit(n_splits=10, val_size=0.1)
    for train_index,val_index in sss.split(X_select,y_select): 
        tag = 0
        X_train,X_val=X[train_index],X[val_index]  
        y_train,y_val=Y[train_index],Y[val_index]
        
        print('train strains: \n')
        for i in range(len(train_index)):
            print(ID_2[train_index[i]],',',end='')
        print('\n')
        
        print('val strains: \n')
        for i in range(len(val_index)):
            print(ID_2[val_index[i]],',',end='')
        y_val_true = np.argmax(y_val,axis = 1) 
        print('val_label: ',y_val_true) 
        print('\n')
        
        
        # RandomForest
        print('--------RandomForest--------')
        treenumber = [600]
        for i in range(6):
            n_estimators_tmp = treenumber[i]
            print('---Treenumer: ---',n_estimators_tmp)
            model = OneVsRestClassifier(RandomForestClassifier(n_estimators=n_estimators_tmp, n_jobs = -1))
            model.fit(X_train, y_train)
            print('RandomForestClassifier: ',model.score(X_val, y_val))
            if model.score(X_val, y_val) > acc:
                acc = model.score(X_val, y_val)
                if tag == 0:
                    tag = 1 
                joblib.dump(model, '/path/to/pkl_dir/'+ drug_name +'_RF_clf.pkl')
               
            y_pred = model.predict(X_val)
                       
            y_pred_true = np.argmax(y_pred,axis = 1) 
            print('predit_label: ', y_pred_true)
            print('confusion_matrix：\n')
            print(confusion_matrix(y_val_true,y_pred_true))      
            RF_fpr,RF_tpr,RF_roc_auc = countROCAUC(y_val,y_RF_score,drug_name,'RandomForest',n_classes)
            print('micro:',RF_roc_auc["micro"])
            print()
        
        # SVM
        print('--------SVM--------')
        clflinear = OneVsRestClassifier(svm.SVC(kernel='linear', C=1, probability=True)).fit(X_train, y_train)
        clfpoly = OneVsRestClassifier(svm.SVC(kernel="poly", degree=5, coef0=1, C=1, probability=True)).fit(X_train, y_train) #多项式：(gamma*u'*v + coef0)^degree
        clfrbf = OneVsRestClassifier(svm.SVC(kernel="rbf", degree=5, coef0=1, C=1, probability=True)).fit(X_train, y_train) #sigmoid：tanh(gamma*u'*v + coef0) 这里是rbf
        
        
        print('SVM-linear: ',clflinear.score(X_val, y_val))
        if clflinear.score(X_val, y_val) > acc or tag == 1:
            acc = clflinear.score(X_val, y_val)
            joblib.dump(clflinear, '/path/to/pkl_dir/'+ drug_name +'_SVMlinear_clf.pkl')
        y_pred = clflinear.predict(X_val)
        y_val_true = np.argmax(y_val,axis = 1)  
        y_pred_true = np.argmax(y_pred,axis = 1) 
        print('predit_label: ', y_pred_true)
        print('confusion_matrix：\n')
        print(confusion_matrix(y_val_true,y_pred_true))
        svmlinear_fpr,svmlinear_tpr,svmlinear_roc_auc = countROCAUC(y_val,y_RF_score,drug_name,'SVM-linear',n_classes)
        print('micro:',svmlinear_roc_auc["micro"])
        
        
        print('SVM-poly: ',clfpoly.score(X_val, y_val))
        y_pred = clfpoly.predict(X_val)
        print('predit_label: ',y_pred)
        y_val_true = np.argmax(y_val,axis = 1)  
        y_pred_true = np.argmax(y_pred,axis = 1) 
        print('predit_label: ', y_pred_true)
        print('confusion_matrix：\n')
        print(confusion_matrix(y_val_true,y_pred_true))
        svmpoly_fpr,svmpoly_tpr,svmpoly_roc_auc = countROCAUC(y_val,y_RF_score,drug_name,'SVM-poly',n_classes)
        print('micro:',svmpoly_roc_auc["micro"])
        if clfpoly.score(X_val, y_val) > acc or tag == 1:
            acc = clfpoly.score(X_val, y_val)
            joblib.dump(clfpoly, '/path/to/pkl_dir/'+ drug_name +'_SVMpoly_clf.pkl')
        
        
        print('SVM-rbf: ',clfrbf.score(X_val, y_val))
        y_pred = clfrbf.predict(X_val)
        y_val_true = np.argmax(y_val,axis = 1)  
        y_pred_true = np.argmax(y_pred,axis = 1) 
        print('predit_label: ', y_pred_true)
        print('confusion_matrix：\n')
        print(confusion_matrix(y_val_true,y_pred_true))
        svmrbf_fpr,svmrbf_tpr,svmrbf_roc_auc = countROCAUC(y_val,y_RF_score,drug_name,'SVM-rbf',n_classes)
        print('micro:',svmrbf_roc_auc["micro"])
        if clfrbf.score(X_val, y_val) > acc or tag == 1:
            acc = clfrbf.score(X_val, y_val)
            joblib.dump(clfrbf, '/path/to/pkl_dir/'+ drug_name +'_SVMrbf_clf.pkl')

        
        # xgboost
        print('--------xgboost--------')
        
        print('---binary:logistic---') 
        model = OneVsRestClassifier(xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=50, 
                silent=True, objective='binary:logistic', n_jobs = -1)) 
        
        model.fit(X_train, y_train)
        y_preds = model.predict(X_test)
        print('predit_label: ',y_preds)
        accuracy = accuracy_score(y_test,y_preds)
        if accuracy > acc or tag == 1:
            acc = accuracy
            joblib.dump(model, '/path/to/pkl_dir/'+ drug_name +'_XGBoost_clf.pkl')
        print("xgboost: " ,accuracy)
        y_test_true = np.argmax(y_test,axis = 1)  
        y_pred_true = np.argmax(y_preds,axis = 1) 
        print('confusion_matrix：\n')
        print(confusion_matrix(y_test_true,y_pred_true))
        XGBoost_fpr,XGBoost_tpr,XGBoost_roc_auc = countROCAUC(y_test,y_XGB_score,drug_name,'XGBoost-binary-logistic',n_classes)
        print('micro:',XGBoost_roc_auc["micro"])
            
    # Draw the ROC curve of the three methods together
    RFmodel = load('/path/to/pkl_dir/'+ drug_name +'_RF_clf.pkl') 
    y_RF_score=RFmodel.predict_proba(X_val)
    print(y_RF_score)
    RF_fpr,RF_tpr,RF_roc_auc = countROCAUC(y_val,y_RF_score,drug_name,'RandomForest',n_classes)
    print('micro:',RF_roc_auc["micro"])
    print()
    
    clflinear = load('/path/to/pkl_dir/'+ drug_name +'_SVMlinear_clf.pkl')
    y_clflinear_score = clflinear.predict_proba(X_val)
    svmlinear_fpr,svmlinear_tpr,svmlinear_roc_auc = countROCAUC(y_val,y_RF_score,drug_name,'SVM-linear',n_classes)
    print('micro:',svmlinear_roc_auc["micro"])
    
    clfpoly = load('/path/to/pkl_dir/'+ drug_name +'_SVMpoly_clf.pkl')
    y_clfpoly_score = clfpoly.predict_proba(X_val)
    svmpoly_fpr,svmpoly_tpr,svmpoly_roc_auc = countROCAUC(y_val,y_RF_score,drug_name,'SVM-poly',n_classes)
    print('micro:',svmpoly_roc_auc["micro"])

    clfrbf = load('/path/to/pkl_dir/'+ drug_name +'_SVMrbf_clf.pkl')
    y_clfrbf_score = clfrbf.predict_proba(X_val)
    #print(y_clfrbf_score)
    svmrbf_fpr,svmrbf_tpr,svmrbf_roc_auc = countROCAUC(y_val,y_RF_score,drug_name,'SVM-rbf',n_classes)
    print('micro:',svmrbf_roc_auc["micro"])

    XGBmodel = load('/path/to/pkl_dir/'+ drug_name +'_XGBoost_clf.pkl')
    y_XGB_score=XGBmodel.predict(X_test)
    XGBoost_fpr,XGBoost_tpr,XGBoost_roc_auc = countROCAUC(y_test,y_XGB_score,drug_name,'XGBoost-binary-logistic',n_classes)
    print('micro:',XGBoost_roc_auc["micro"])

    
    #print picture
    plt.figure()
    lw = 2
    plt.plot(RF_fpr["micro"], RF_tpr["micro"],
             label='RandomForest ROC curve (area = {0:0.2f})'
                   ''.format(RF_roc_auc["micro"]),
             color='deeppink', linewidth=2)

    
    plt.plot(svmlinear_fpr["micro"], svmlinear_tpr["micro"],
             label='SVM(linear) ROC curve (area = {0:0.2f})'
                   ''.format(svmlinear_roc_auc["micro"]),
             color='navy', linewidth=2)

    
    plt.plot(svmpoly_fpr["micro"], svmpoly_tpr["micro"],
             label='SVM(poly) ROC curve (area = {0:0.2f})'
                   ''.format(svmpoly_roc_auc["micro"]),
             color='aqua', linewidth=2)

    
    plt.plot(svmrbf_fpr["micro"], svmrbf_tpr["micro"],
             label='SVM(rbf) ROC curve (area = {0:0.2f})'
                   ''.format(svmrbf_roc_auc["micro"]),
             color='darkorange', linewidth=2)

    plt.plot(XGBoost_fpr["micro"], XGBoost_tpr["micro"],
             label='XGBoost ROC curve (area = {0:0.2f})'
                   ''.format(XGBoost_roc_auc["micro"]),
             color='cornflowerblue', linewidth=2)
    
    
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 5,
    }

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating curve')
    plt.legend(loc="lower right",prop=font1)
    plt.savefig(drug_name+'-ROC-AUC-all.pdf')    
     		


def drugtest():
    global X
    trainmodel(X,Y_MIC,'your-drug-name')

drugtest()
