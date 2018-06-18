import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import networkx as nx
import time
import math
import os

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
import warnings
from katz import * 
warnings.filterwarnings('ignore')

#clasificacion
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB    #Naive Bayes
from sklearn.neighbors import KNeighborsClassifier #knn
from sklearn.neural_network import MLPClassifier #MLP
from sklearn.svm import SVC #SVM
from sklearn.tree import DecisionTreeClassifier #decision tree



#evaluacion de resultados
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def minimizedCN(H,nodeH, dataTest):
    df = []
    dfTest = pd.DataFrame(data = dataTest, columns = ['N1','N2'])
    cont1 = 0
    cont2 = 0
    kz = katz(H)
    for n in nodeH:
        cont2 = 0
        for n2 in nodeH:
            if H.has_edge(n,n2) == True or n == n2 or nodeH.index(n) >= nodeH.index(n2) or len(sorted(nx.common_neighbors(H, n, n2))) == 0 or sorted(nx.adamic_adar_index(H, [(n, n2)]))[0][2] is 0 or kz[cont1,cont2] == 0:
                cont2 = cont2 + 1
            else:
                d = {'N1': [n], 'N2': [n2]}
                d1 = {'N1': [n2], 'N2': [n]}
                df2 = pd.DataFrame(data=d)
                df3 = pd.DataFrame(data=d1)
                if len(df2.merge(dfTest, how='inner', on = ['N1','N2'])) > 0:
                    df.append([str(n) + "-" + str(n2) ,len(list(nx.common_neighbors(H, n, n2))),sorted(nx.adamic_adar_index(H, [(n, n2)]))[0][2], kz[cont1,cont2], 1])
                elif len(df3.merge(dfTest, how='inner', on = ['N1','N2'])) > 0:
                    df.append([str(n) + "-" + str(n2) ,len(list(nx.common_neighbors(H, n, n2))),sorted(nx.adamic_adar_index(H, [(n, n2)]))[0][2], kz[cont1,cont2], 1])
                else:
                    df.append([str(n) + "-" + str(n2), len(list(nx.common_neighbors(H, n, n2))),sorted(nx.adamic_adar_index(H, [(n, n2)]))[0][2], kz[cont1,cont2], 0])
                cont2 = cont2 + 1
        cont1 = cont1 + 1
    return df



def Testing(loops,address,debugger,text): 
    text.write("DataTest: " + address + "\n") 
    text.write("================================================" + "\n") 
    if(debugger):
        print("Leyendo data de ", address)
    dataTest = pd.read_csv(address, delim_whitespace=True, header=None, names=['N1','N2'])
    dataTest = pd.DataFrame(data = dataTest)
    miData = pd.read_csv(address, delim_whitespace=True, header=None, names=['N1','N2'])
    DataE, DataT = train_test_split(miData,random_state = 22, test_size = 0.2)
    H = nx.from_pandas_dataframe(DataE,'N1','N2')
    nodeH = H.nodes(data=False)    
    
    
    #==========================Common N======================================
    
    

    if(debugger):
        print("Corriendo CD")
    for n in range(loops):
        if(debugger):
            print("Loop de CN ", n + 1)
        df = minimizedCN(H,nodeH,dataTest)
        labels = ['Nodos', 'CN', 'AD','KZ', 'Amigos']
        dfFinal = pd.DataFrame.from_records(df, columns=labels)
        
        X = dfFinal.iloc[:,1:4]
        Y = dfFinal.iloc[:,4]
        rescaledX = StandardScaler().fit_transform(X)
        Xnuevo = pd.DataFrame(data = rescaledX, columns= X.columns)
        
        X_train,X_test,Y_train,Y_test = train_test_split(Xnuevo,Y, test_size = 0.3)
        models = []
        models.append(("MLP 5x2", MLPClassifier(solver='lbfgs', hidden_layer_sizes=(5, 2),  max_iter=10)))  #instancia MLP
        
        results = []
        names = []
        for name,model in models:
            kfold = KFold(n_splits=10, random_state=22) #k-fold validation
            cv_result = cross_val_score(model,X_train,Y_train, cv = kfold,scoring = "accuracy")
            names.append(name)
            results.append(cv_result)
        for i in range(len(names)):
            print(names[i],results[i].mean())


def main():
    fileDirs = [line.rstrip() for line in open('Files1.txt')]
    if not os.path.exists("./Results"):
        os.makedirs("./Results")
    i = 1
    for dir in fileDirs:
        print(dir)
        file = open("./Results/Data" + str(i) + ".txt","w") 
        Testing(1,dir,True,file)
        i += 1
        file.close() 
    pass

if __name__ == "__main__":
    main()
    