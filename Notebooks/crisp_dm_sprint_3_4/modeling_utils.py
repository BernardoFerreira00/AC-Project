# Functions to help with data understanding 

from imblearn import under_sampling, over_sampling
from sklearn.model_selection import train_test_split, StratifiedKFold

def split_data(X, y, test_size):
    return train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

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
        
        
        
 