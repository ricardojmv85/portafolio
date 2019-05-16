
# PORFAFOLIO DE FUNCIONES 

#El siguiente portafolio contiene funciones que facilitan el analicis y el orden en el codigo a la hora de ejecutar funciones de machine learning.

# funcion que importa las librerias basicas para manejar dataframes, graficas y tener funciones de metricas de sklearn.
def importaciones_basicas():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    import pandas as pd
    import seaborn as sns
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    from sklearn.model_selection import cross_val_score
    from __future__ import division
    from sklearn.utils import shuffle

#######################################################################################################################################################################################
# funcion que importa las librerias basicas para correr redes neuronales.
def importaciones_tensorflow():
    import tensorflow as tf
    from tensorflow.keras.layers import Dropout
    from tensorflow import keras
    from tensorflow.keras import layers
    from keras.models import load_model
    from tensorflow.keras.layers import Dense, Flatten, Activation 

#######################################################################################################################################################################################
    
# funcion que importa las librerias necesarias para ejecutar los algoritmos de clusterizacion de sklearn
def importaciones_clustering():
    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.cluster import MeanShift, estimate_bandwidth
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from collections import Counter
    from sklearn.metrics import confusion_matrix

#######################################################################################################################################################################################

# funcion que importa las librerias necesarias de sklearn para ejecutar arboles de decision.
def importaciones_arboles():
    from sklearn.utils.multiclass import unique_labels
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor

#######################################################################################################################################################################################

# funcion que recibe un dataframe y retorna las caracteristicas principales en una lista -> [las dimensiones del dataframe, el nombre de las columnas,el head del dataframe, el tail del dataframe]
def caracteristicas_principales(df):
    return([df.shape,df.columns.values,df.head(),df.tail()])

#######################################################################################################################################################################################

#funcion que recibe el dataframe y una lista de columnas por las cuales quiere eliminar las observaciones repetidas 
def eliminar_repetidos(df,columns):
    return(df.drop_duplicates(subset=columns))

#######################################################################################################################################################################################

# funcion que recibe el dataframe para obtener los componentes principales y el numero de componentes principales.
def pca(df,componentes):
    import sklearn
    from sklearn.decomposition import PCA
    pca=PCA(n_components=componentes)
    X_pca2= pca.fit_transform(df)
    return(pd.DataFrame(X_pca2))

#######################################################################################################################################################################################

# funcion que recibe el dataframe que contiene los datos, y una lista de los nombres de las dos variables a graficas.
def grafica_barras(df,columnas):
    return(sns.barplot(x=columnas[0],y=columnas[1], 
    data=df, color="c"))

#######################################################################################################################################################################################

# funcion que recibe un dataframe, y una lista de los nombres de las columnas para graficar scatter plot. y el nombre de la columa por la cual se diferencia el color en un cluster
def graph1(df,columnas):
    colors= ["#e74c3c","#3C64E7"]
    sns.lmplot(x=columnas[0], y=columnas[1],
        data=df, 
        fit_reg=False, 
        hue='Cluster', # color por cluster
        legend=True,
        scatter_kws={"s": 150},
        markers=["o", "o"],palette=colors)

#######################################################################################################################################################################################

#funcion que toma un dataframe y una lista con las 
def mean_shift_grafica(x):
    colors= ["#3C64E7","#e74c3c"]
    sns.lmplot(x="firstInhibitor", y="towerKills",
        data=x, 
        fit_reg=False, 
        hue='Cluster', # color by cluster
        legend=True,
        scatter_kws={"s": 150},
        markers=["o"],palette=colors)

#######################################################################################################################################################################################

# funcion que recibe un modelo svc, el nombre del kernel, el dataset de entrenamiento y el de test y retorna una grafica del hiperplano dividiendo la data
def plotSVC(svc, param, X, y):
    clf = svc
    clf.fit(X, y)

    plt.clf()
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                edgecolor='k', s=20)

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none',
                zorder=10, edgecolor='k')

    plt.axis('tight')
    x_min = X.iloc[:, 0].min()
    x_max = X.iloc[:, 0].max()
    y_min = X.iloc[:, 1].min()
    y_max = X.iloc[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    pre_z = svc.predict(np.c_[XX.ravel(), YY.ravel()])

    Z = pre_z.reshape(XX.shape)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                linestyles=['--', '-', '--'])
    
    plt.pcolormesh(XX, YY, Z , cmap=plt.cm.Paired)
    plt.title(param)
    plt.show()

#######################################################################################################################################################################################

#funcion que ejecuta una regresion logistica, tomando como parametros un test de x y y , la funcion patte la data en entremiento y retorna el accuracy de la prediccion en el test set.
def lr(x,y):
    X_train, X_test,y_train,y_test = train_test_split(x,y, test_size=0.25, random_state=42)
    clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='ovr').fit(X_train, y_train.values.ravel())
    compare(clf.predict(X_test),y_test.values.ravel())
    return clf.predict(X_test),X_test

#######################################################################################################################################################################################

# funcion que ejecuta un analizador discriminador cuadratico, recibiendo los dataframes de features y target variable, la funcion ejecuta la particion de la data y ajusta qda a la data de entreamiento, y retorna el accuracy dell test
def qda(x,y):
    X_train, X_test,y_train,y_test = train_test_split(x,y, test_size=0.25, random_state=100)
    print "QDA"
    clf = QuadraticDiscriminantAnalysis().fit(X_train,y_train.values.ravel())
    compare(clf.predict(X_test),y_test.values.ravel())
    return clf.predict(X_test),X_test

#######################################################################################################################################################################################

# funcion que ejecuta un analizador discriminador lineal, recibiendo los dataframes de features y target variable, la funcion ejecuta la particion de la data y ajusta qda a la data de entreamiento, y retorna el accuracy dell test
def lda(x,y):
    X_train, X_test,y_train,y_test = train_test_split(x,y, test_size=0.25, random_state=100)
    clf = LinearDiscriminantAnalysis().fit(X_train,y_train.values.ravel())
    compare(clf.predict(X_test),y_test.values.ravel())
    return clf.predict(X_test),X_test

#######################################################################################################################################################################################

# funcion que recibe dos dataframe, uno con las fesatures y el otro con la repsonse variable. La funcion retorna el accuracy de las diferentes hirarchical clustering utlizando diferentes linkage methods.
def hca(x,y):
    for linkage in ('ward', 'average', 'complete', 'single'):
        clustering = AgglomerativeClustering(linkage=linkage, n_clusters=2).fit(x)
        print(linkage)
        compare(clustering.labels_,y.values.ravel())
        print ""

#######################################################################################################################################################################################

# Esta funcion compara los resultados predecidos con los reales y retorna el accuracy, recibe dos listas en total.
def compare(cluster_labels,real_labels):
    quantity=len(cluster_labels)
    count=0
    for i in range(0,quantity):
        if(cluster_labels[i]==real_labels[i]):
            count=count+1
    print "Acurracy: ",round(count/quantity,5)*100

#######################################################################################################################################################################################

# funcion que recibe dos dataframe, uno con las fesatures y el otro con la repsonse variable. La funcion retorna el accuracy del expectation maximization
def em(x,y):
    gmm = GaussianMixture(n_components=2, covariance_type='full').fit(x)
    compare(gmm.predict(x),y.values.ravel())
    return gmm.predict(x)

#######################################################################################################################################################################################

# funcion que recibe dos dataframe, uno con las fesatures y el otro con la repsonse variable. La funcion retorna el accuracy del mean shift clustering
def ms(x,y):
    bandwidth = estimate_bandwidth(x, quantile=0.5)
    clustering = MeanShift(bandwidth=bandwidth).fit(x)
    compare(clustering.labels_,y.values.ravel())
    return clustering.labels_

#######################################################################################################################################################################################

# funcion que recibe dos dataframe, uno con las fesatures y el otro con la repsonse variable. La funcion retorna el accuracy del kmeans con n clusters.
def km(x,y,n):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(x)
    compare(abs(kmeans.labels_-1),y.values.ravel())
    return X,abs(kmeans.labels_-1)

#######################################################################################################################################################################################

# funcion que recibe una lista con las features independientes y la data para ajustar una regresion lineal simple y retorna el R cuadrado por cada variable como predictora
def SLR(data,feas):
    results=[]
    for fea in feas:
        #fit the regresssion
        res = sm.OLS(data[["critical_temp"]],data[fea]).fit()
        #insert the results this temporal array
        tmp=[fea,res.rsquared,res.rsquared_adj,res.fvalue,res.f_pvalue]
        #convert the array into a dataFrame
        results.append(tmp)
    results=pd.DataFrame(data=results,columns=['feature','R2','Adj(R2)','F value','F p value'])
    #return dataframe with the results
    return results.sort_values('R2',ascending=0)

#######################################################################################################################################################################################

# funcion que recibe una lista con las features independientes y la data para ajustar una regresion lineal multiple y retorna el R cuadrado del modelo
def LR(data,feas,response):
    #fit the regresssion
    res = sm.OLS(data[[response]],data[feas]).fit()
    #insert the results this temporal array
    tmp=[['ALL',res.rsquared,res.rsquared_adj,res.fvalue,res.f_pvalue]]
    #convert the array into a dataFrame
    tmp=pd.DataFrame(data=tmp,columns=['feature','R2','Adj(R2)','F value','F p value'])
    #return dataframe with the results
    return tmp.sort_values('R2',ascending=0)

#######################################################################################################################################################################################

# funcion que recibe un dataframe, lista de features para ajustar la regresion polinomial, la variable response y el grado del polinomio
def PL(data,feas,deg):
    #transforms the features to polynomial
    poly_reg = PolynomialFeatures(degree = deg)
    X_poly = poly_reg.fit_transform(pd.DataFrame(data[feas]))
    #fit the model with the tranformed features
    mod = sm.OLS(data[["critical_temp"]],X_poly)
    res=mod.fit()
    #list of the models with R2, ADJ(R2), F value and P value for the F-statistic
    tmp=[['ALL',res.rsquared,res.rsquared_adj,res.fvalue,res.f_pvalue]]
    tmp=pd.DataFrame(data=tmp,columns=['feature','R2','Adj(R2)','F value','F p value'])
    return tmp.sort_values('R2',ascending=0)
pd.set_option('display.max_rows', 100)

#######################################################################################################################################################################################

# funcion que ajuste e imprime las metricas principales de una regresion ridge, los dataframes de X y Y, ademas de alpha que es el coeficiente de penalizacion.
def ridge_regresion(,y,alpha2):
    X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.25, random_state=100)
    ridge_model = Ridge(alpha=alpha2, fit_intercept=True)
    ridge_model.fit(X_train, y_train)
    print "Ridge Regression"
    print "R2: ", ridge_model.score(X_test, y_test)
    print "Coefficients: " ,ridge_model.coef_[0][:10]

#######################################################################################################################################################################################

# funcion que ajusta e imporime als metricas princiaples de una regresion lasso, recibe los dataframes de X y Y, ademas del parametro de regularizacion.
def laddo_regresion(X,y,alpha2):
    X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.25, random_state=100)
    print "Lasso Regression"
    lasso_model_0 = Lasso(alpha=alpha2, fit_intercept=True)
    lasso_model_0.fit(X_train, y_train)
    print "R2 Lasso: ", lasso_model_0.score(X_test, y_test)
    print "Coefficients Lasso: " ,lasso_model_0.coef_[:10]

#######################################################################################################################################################################################

# funcion que recibe dos dataframes, X_train, y_train, y retonra una grafica que muestra el accuracy mientras aumenta la profundidad del arbol de decision.
def arbol_altura(X_train,y_train):
    scores=[]
    depth=[]
    for i in range(2,51):
        clf = DecisionTreeClassifier(max_depth=i)
        clf = clf.fit(X_train, y_train)
        depth.append(i);
        scores.append(np.mean(cross_val_score(clf, X_train, y_train, cv=10)))
    score=pd.DataFrame()
    score['scores']=scores
    score['depth']=depth

    print score.loc[score['scores']==max(score['scores'])]
    fig=plt.figure(figsize=(16,6))
    ax=fig.add_subplot(1,2,1)
    ax.set_xlabel('Depth')
    ax.set_ylabel('Score')
    ax.set_title('Depth vs Score')
    ax.plot(score.depth,score.scores,color='r')
    ax.legend()

#######################################################################################################################################################################################

# funcion que recibe dos dataframe X_train y y_train, ejecuta arboles de decision y retorna una funcion que muestra la relacion entre accuracy y la cantiad de observaciones minimas para esplitear un nodo.
def arbol_hojas(X_train,y_train):
    scores=[]
    min_samples=[]
    for i in range(2,51):
        clf = DecisionTreeClassifier(min_samples_leaf=i)
        clf = clf.fit(X_train, y_train)
        min_samples.append(i);
        scores.append(np.mean(cross_val_score(clf, X_train, y_train, cv=10)))
    score=pd.DataFrame()
    score['scores']=scores
    score['min_samples']=min_samples
        
    print score.loc[score['scores']==max(score['scores'])]
    fig=plt.figure(figsize=(16,6))
    ax=fig.add_subplot(1,2,1)
    ax.set_xlabel('Min samples')
    ax.set_ylabel('Score')
    ax.set_title('Min Samples vs Score')
    ax.plot(score.min_samples,score.scores,color='r')
    ax.legend()

#######################################################################################################################################################################################

# funcion que recibe X_train y y_train dataframe y retorna una grafica que muestra la relacion entre accuracy y el numero de estimadores en los random forests para este dataframe.
def random_forest_estimadores(X_train,y_train):
    scores=[]
    estimators=[25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500,525,550,575,600]
    for i in estimators:
        clf = RandomForestClassifier(n_estimators=i, n_jobs=-1,oob_score=True)
        clf.fit(X_train, y_train)
        scores.append(clf.oob_score_)
    score=pd.DataFrame()
    score['scores']=scores
    score['estimators']=estimators

    print score.loc[score['scores']==max(score['scores'])]
    fig=plt.figure(figsize=(16,6))
    ax=fig.add_subplot(1,2,1)
    ax.set_xlabel('Estimator')
    ax.set_ylabel('Score')
    ax.set_title('Estimators vs Score')
    ax.plot(score.estimators,score.scores,color='r')
    ax.legend()

#######################################################################################################################################################################################

# funcion que recibe X_train, y_train y retorna una grafica con la relacion entre accuracy y el numero de estimadores utilizados en boosting.
def adaboost_estimadores(X_train,y_train):
    scores=[]
    estimators=[25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500,525,550,575,600]
    for i in estimators:
        AdaBoost = AdaBoostClassifier(n_estimators=i,learning_rate=1,algorithm='SAMME')
        AdaBoost.fit(X_train,y_train)
        scores.append(np.mean(cross_val_score(AdaBoost, X_train, y_train, cv=10)))
    score=pd.DataFrame()
    score['scores']=scores
    score['estimators']=estimators

    print score.loc[score['scores']==max(score['scores'])]
    fig=plt.figure(figsize=(16,6))
    ax=fig.add_subplot(1,2,1)
    ax.set_xlabel('Estimators')
    ax.set_ylabel('CV Score')
    ax.set_title('Estimators vs Score')
    ax.plot(score.estimators,score.scores,color='r')
    ax.legend()

#######################################################################################################################################################################################

# funcion que recibe X_train, y_train y retorna una grafica con la relacion entre accuracy y el numero de estimadores utilizados en gradient boosting.
def gradientboos_estimadores():
    scores=[]
    estimators=[25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500,525,550,575,600]
    for i in estimators:
        Gradient = GradientBoostingClassifier(n_estimators=i)
        Gradient.fit(X_train,y_train)
        scores.append(np.mean(cross_val_score(Gradient, X_train, y_train, cv=10)))
    score=pd.DataFrame()
    score['scores']=scores
    score['estimators']=estimators

    print score.loc[score['scores']==max(score['scores'])]
    fig=plt.figure(figsize=(16,6))
    ax=fig.add_subplot(1,2,1)
    ax.set_xlabel('Estimators')
    ax.set_ylabel('CV Score')
    ax.set_title('Estimators vs Score')
    ax.plot(score.estimators,score.scores,color='r')
    ax.legend()

########################################################################################################################################################################################

# funcion que grafica una matriz de confusion recibiendo como parametros los labels reales y los labels predecidos en forma de lista.
def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


##############################################################################################################################################################################3

# funcion que convierte las imagenes de un directorio en arrays que representan las imagenes en blanco y negro y con una dimension mas baja,
def image_preprocessing():
    X = [] # the images 
    y=[] # the labels of classes of flowers
    DATADIR="flowers-recognition/flowers/" #directory to the images folders
    IMG_SIZE=60 #sizse of the images
    CATEGORIES=[] 

    # this function will insert the name of each folder in the CATEGORIES array for future navigation
    for i in os.listdir(DATADIR):
        CATEGORIES.append(i)

    class_num=0
    for category in CATEGORIES: # fot each category or folder in the image dataset
        path = os.path.join(DATADIR,category) #making the path to an specific flower class folder

        num_images=0
        
        # for the first 700 images in the folder it will transform the image to an array, then it will reshape 
        for img in (os.listdir(path)):
            if(num_images<700):
                try:
                    img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE) #transform the image to array
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # reshape the image
                    X.append(new_array)  #adding the image array to the X array of images
                    y.append(class_num) # adding the class of flower of the image to the laberls array
                    num_images+=1
                except Exception as e:
                        True
            else:
                break
        class_num+=1
            
