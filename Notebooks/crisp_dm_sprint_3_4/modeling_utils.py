# Functions to help with data understanding 

from imblearn import under_sampling, over_sampling
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor
def split_data(X, y, test_size=0.3):
    return train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)

def smote_over_sampling(X_train, y_train,option="",rs=42):
    if(option=="SMOTE"):
        sm = over_sampling.SMOTE(random_state=rs)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    elif(option=="ADASYN"):
        adn = over_sampling.ADASYN(random_state=rs)
        X_train, y_train = adn.fit_resample(X_train, y_train)
    elif(option=="BorderlineSMOTE"):
        blst = over_sampling.BorderlineSMOTE(random_state=rs)
        X_train, y_train = blst.fit_resample(X_train, y_train)
    elif(option=="SVMSMOTE"):
        svmte = over_sampling.SVMSMOTE(random_state=rs)
        X_train, y_train = svmte.fit_resample(X_train, y_train)
    elif(option=="KMeansSMOTE"):
        kmsmt = over_sampling.KMeansSMOTE(random_state=rs)
        X_train, y_train = kmsmt.fit_resample(X_train, y_train)
    elif(option=="RandomOverSampler"):
        ros = over_sampling.RandomOverSampler(random_state=rs)
        X_train, y_train = ros.fit_resample(X_train, y_train)
    elif(option=="SMOTENC"):
        smtc = over_sampling.SMOTENC(random_state=rs)   
        X_train, y_train = smtc.fit_resample(X_train, y_train)
    elif(option=="SMOTEN"):
        smtn = over_sampling.SMOTEN(random_state=rs)
        X_train, y_train = smtn.fit_resample(X_train, y_train)
        
        
def smote_under_sampling(X_train, y_train,option="",rs=42):
    if(option=="RandomUnderSampler"):
        rus = under_sampling.RandomUnderSampler(random_state=rs)
        X_train, y_train = rus.fit_resample(X_train, y_train)
    elif(option=="TomekLinks"):
        tl = under_sampling.TomekLinks(random_state=rs)
        X_train, y_train = tl.fit_resample(X_train, y_train)
    elif(option=="NearMiss"):
        nm = under_sampling.NearMiss(random_state=rs)
        X_train, y_train = nm.fit_resample(X_train, y_train)
    elif(option=="CondensedNearestNeighbour"):
        cnn = under_sampling.CondensedNearestNeighbour(random_state=rs)
        X_train, y_train = cnn.fit_resample(X_train, y_train)
    elif(option=="OneSidedSelection"):
        oss = under_sampling.OneSidedSelection(random_state=rs)
        X_train, y_train = oss.fit_resample(X_train, y_train)
    elif(option=="NeighbourhoodCleaningRule"):
        ncr = under_sampling.NeighbourhoodCleaningRule(random_state=rs)
        X_train, y_train = ncr.fit_resample(X_train, y_train)
    elif(option=="InstanceHardnessThreshold"):
        iht = under_sampling.InstanceHardnessThreshold(random_state=rs)
        X_train, y_train = iht.fit_resample(X_train, y_train)
    elif(option=="EditedNearestNeighbours"):
        enn = under_sampling.EditedNearestNeighbours(random_state=rs)
        X_train, y_train = enn.fit_resample(X_train, y_train)
    elif(option=="RepeatedEditedNearestNeighbours"):
        reenn = under_sampling.RepeatedEditedNearestNeighbours(random_state=rs)
        X_train, y_train = reenn.fit_resample(X_train, y_train)
    elif(option=="AllKNN"):
        aknn = under_sampling.AllKNN(random_state=rs)
        X_train, y_train = aknn.fit_resample(X_train, y_train)
    elif(option=="NeighbourhoodCleaningRule"):
        ncr = under_sampling.NeighbourhoodCleaningRule(random_state=rs)
        X_train, y_train = ncr.fit_resample(X_train, y_train)
    elif(option=="ClusterCentroids"):
        cc = under_sampling.ClusterCentroids(random_state=rs)
        X_train, y_train = cc.fit_resample(X_train, y_train)      


# Blocking time series cross validation

class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits
    
    def get_n_splits(self, X, y, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]


def build_model(_alpha, _l1_ratio):
    estimator = ElasticNet(
        alpha=_alpha,
        l1_ratio=_l1_ratio,
        fit_intercept=True,
        normalize=False,
        precompute=False,
        max_iter=16,
        copy_X=True,
        tol=0.1,
        warm_start=False,
        positive=False,
        random_state=None,
        selection='random'
    )

    return MultiOutputRegressor(estimator, n_jobs=4)



#TODO:
# ver melhor como integrar isto em funções 
#Adicionar cross-validation ideal para time series
        
# def calculate_statistics(y_test, pred,result_dict):
#     acc_list = []
#     auc_list = []
#     cm_list = []
#     accuracy = metrics.accuracy_score(y_test, pred)
#     precision = metrics.precision_score(y_test, pred)
#     recall = metrics.recall_score(y_test, pred)
#     f1_score1 = metrics.f1_score(y_test, pred)
#     acc_list.append(metrics.accuracy_score(y_test, pred))
#     fpr, tpr, _thereshold = metrics.roc_curve(y_test, pred)
#     auc_list.append(round(metrics.auc(fpr, tpr),3))
#     cm_list.append(metrics.confusion_matrix(y_test, pred))
#     result_dict = {"accuracy":accuracy,"precision":precision,"recall":recall,"f1_score":f1_score1,"auc":auc_list,"confusion_matrix":cm_list,"acc_list":acc_list}

#     print(f"Accuracy: {accuracy}")
#     print(f"Precision: {precision}")
#     print(f"Recall: {recall}")
#     print(f"F1-Score: {f1_score1}")
    
    
    
# def add_model_to_pipeline(model,name_model ,model_pipeline):
#     model_pipeline[name_model] = model
    

# # save prediction on model_array of dataframes predictions and then save it to csv
# def model_pipeline_evaluation(model_pipeline, X_train, y_train, X_test, y_test):
#     result_dict = {}
#     for name_model, model in model_pipeline.items():
#         print(name_model)
#         model.fit(X_train, y_train)
#         pred = model.predict(X_test)
#         calculate_statistics(y_test, pred,result_dict)
#         print(
    