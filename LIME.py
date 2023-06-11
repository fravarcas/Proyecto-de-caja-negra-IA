import numpy as np
import scipy as sc
import pandas as pd
import sklearn as sk
from sklearn.metrics import precision_score, roc_auc_score
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn import svm
from scipy.stats import spearmanr
from tensorflow import keras
from keras.utils import to_categorical

#carga de datos
adults = pd.read_csv('proyecto_caja_negra/adult.data', header=None,
                       names=['age', 'workclass', 'fnlwgt', 'education',
                              'education-num', 'marital-status', 'occupation', 'relationship',
                                'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'money'])

attributes_adults = adults.loc[:, 'age':'native-country']
goal_adult = adults['money']

codificator_attributes = sk.preprocessing.OrdinalEncoder()
codificator_attributes.fit(attributes_adults)
codified_attributes = codificator_attributes.transform(attributes_adults)

codificator_goal = sk.preprocessing.LabelEncoder()
codified_goal = codificator_goal.fit_transform(goal_adult)

max_attributes_adults = [max(codified_attributes[:, x]) for x in range(14)]
min_attributes_adults = [min(codified_attributes[:, x]) for x in range(14)]

(training_attributes, test_attributes,
 training_goal, test_goal) = model_selection.train_test_split(
     codified_attributes,codified_goal, random_state=12345, test_size=.33, stratify=codified_goal)


poker_hands = pd.read_csv('proyecto_caja_negra/poker-hand-testing.data', header=None,
                          names=['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'class'])

attributes_poker = poker_hands.loc[:, 'S1':'C5']
goal_poker = poker_hands['class']

max_attributes_poker = [max(codified_attributes[:, x]) for x in range(10)]
min_attributes_poker = [min(codified_attributes[:, x]) for x in range(10)]


(training_attributes_poker, test_attributes_poker,
 training_goal_poker, test_goal_poker) = model_selection.train_test_split(attributes_poker, goal_poker, random_state=12345, test_size=.33, stratify=goal_poker)

#Entrenamiento de modelo random forest model para datos adult
randomForestModel = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
randomForestModel.fit(training_attributes, training_goal)

#Entrenamiento de modelo xgboost para datos adult

training_data = xgb.DMatrix(training_attributes, label= training_goal)

#Se definen los parametros para una tarea de clasificación binaria
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
}

xgboostModel = xgb.train(params, training_data)

#Entrenamiento de modelo redes neuronales para datos poker_hands
'''
attributes_neural_network = attributes_poker.to_numpy()
goal_neural_network = goal_poker.to_numpy()

classes = 10
codified_goal_poker = to_categorical(goal_neural_network, classes)

(training_attributes_neural, test_attributes_neural,
 training_goal_neural, test_goal_neural) = model_selection.train_test_split(attributes_neural_network, codified_goal_poker, random_state=12345, test_size=.33, stratify=codified_goal_poker)

normalizador = keras.layers.Normalization()
normalizador.adapt(training_attributes_neural)

poker_hand = keras.Sequential()
poker_hand.add(keras.Input(shape=(10,)))
poker_hand.add(normalizador)
poker_hand.add(keras.layers.Dense(12))
poker_hand.add(keras.layers.Dense(12))
poker_hand.add(keras.layers.Dense(10, activation='softmax'))
poker_hand.compile(optimizer='SGD', loss='categorical_crossentropy')

poker_hand.fit(training_attributes_neural, training_goal_neural,
                batch_size=256, epochs=5)
'''
#Entrenamiento de modelo xgboost para datos poker_hands

training_data_poker = xgb.DMatrix(training_attributes_poker, label= training_goal_poker)

#Se definen los parametros para una tarea de clasificación multiclase
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'multi:softmax',
    'eval_metric': 'logloss',
    'num_class': '10'
}

xgboostModel = xgb.train(params, training_data)


#función para calcular la distancia coseno de dos muestras
def cosine_distance(muestra1, muestra2):

    vector1 = np.array(muestra1)
    vector2 = np.array(muestra2)

    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    cosine_distance = dot_product / (norm1 * norm2)

    return cosine_distance


def LIMEAlgorithm(data, f, N, max_attributes, min_attributes):

    #Inicalizamos las listas donde se guardaran los diferentes conjuntos
    X = []
    R = []
    W = []
    Y = []

    #Realizamos N iteraciones para generar N permutaciones de los atributos de la muestra
    for i in range (N):

    #Generamos una lista aleatoria que tenga como maximo el tamaño de la muestra y cuyos enteros esten entre 0 y el tamaño de la muestra
        attributes = np.random.choice(len(data),np.random.randint(len(data)), replace=False)
    #Generamos las sublistas que representaran los diferentes valores para 1 permutación de la muestra
        element_R = []
        element_X = []
    #Recorremos los atributos de la muestra
        for j in range(len(data)):
    #Comprobamos si el indice del atributo está en la lista generada aleatoriamente para decidir si perturbamos el atributo o no
            if j in attributes:

                element_R.append(0)
                modifiedAttribute = np.random.randint(min_attributes[j], max_attributes[j])
                element_X.append(modifiedAttribute)

            else:

                element_R.append(1)
                element_X.append(data[j])
            

        W.append(cosine_distance(element_X, data))

        X.append(element_X)
        R.append(element_R)
    #Una vez realizadas las permutaciones las ponderamos con los pesos generados en W
    R_ponderada = []

    for i in range(len(R)):

        num = W[i]
        sublist = R[i]
        R_muestra_ponderada = []

        for j in range(len(sublist)):

            res = num * sublist[j]
            R_muestra_ponderada.append(res)

        R_ponderada.append(R_muestra_ponderada)

    #Realizamos las predicciones de las permutaciones con el modelo pasado como parametro
    if f == randomForestModel:
        for i in range(len(X)):

            prediction = f.predict(np.array(X[i]).reshape(1, -1))
            Y.append(prediction)

    elif f == xgboostModel:
        for i in range(len(X)):
            test = np.array(X[i]).reshape(1, -1)
            test_xgb = xgb.DMatrix(test)
            prediction = f.predict(test_xgb)
            Y.append(prediction)
    #Se entrena el modelo ridge pasandole las R ponderadas como atributos y las predicciones de Y como objetivo
    G = Ridge()
    G.fit(R_ponderada, Y)

    return G.coef_, G.intercept_, G

#Metricas

def identidad(muestra_1, muestra_2, f, max_attribute,  min_attribute):

    d_1, a, b = LIMEAlgorithm(muestra_1, f, 1000, max_attribute, min_attribute)
    d_2, a, b = LIMEAlgorithm(muestra_2, f, 1000, max_attribute, min_attribute)

    distancia_muestras = cosine_distance(muestra_1, muestra_2)

    if distancia_muestras == 1.0:

        distancia_explicación = cosine_distance(np.squeeze(np.asarray(d_1)), np.squeeze(np.asarray(d_2)))

        return distancia_explicación

    else:

        print('estas muestras no son identicas')
        
        
        
def separabilidad(muestra_1, muestra_2, f, max_attribute,  min_attribute):

    d_1, a, b = LIMEAlgorithm(muestra_1, f, 100, max_attribute, min_attribute)
    d_2, a, b = LIMEAlgorithm(muestra_2, f, 100, max_attribute, min_attribute)


    distacia_muestras_ab = cosine_distance(muestra_1, muestra_2)

    if distacia_muestras_ab != 1.0:
        
        distancia_explicación = cosine_distance(np.squeeze(np.asarray(d_1)), np.squeeze(np.asarray(d_2)))
    
        return distancia_explicación

    else:

        print("estas muestras son identicas por lo tanto no se puede comprobar separabilidad")

def estabilidad(explicaciones_similares, explicaciones_diferentes):
     
    matriz_similares = np.array(explicaciones_similares)
    matriz_diferentes = np.array(explicaciones_diferentes)
        
    correlacion, _ = spearmanr(matriz_similares, matriz_diferentes, axis=1)
    
    return correlacion
       
def selectividad(test_attribute, test_goal, f, orden):

    y_pred_orig = f.predict(test_attribute)
    auc_orig = roc_auc_score(test_goal, y_pred_orig)
    auc_values = []

    for i in orden:

        attribute_test_modified = test_attribute.copy()
        attribute_test_modified[:, i] = 0 
        y_pred_modified = f.predict(attribute_test_modified)
        auc_modified = roc_auc_score(test_goal, y_pred_modified)
        auc_change = auc_orig - auc_modified
        auc_values.append(auc_change)

    selectivity_scores = auc_values

    return selectivity_scores

        
def coherencia(p, e):
    diferencia=abs(p-e)
    return diferencia


def completitud(muestras, G, f, test):
    
    numero_de_muestras = muestras.shape[0]
    error_explicacion = 0
    error_prediccion = 0

    for i in range(numero_de_muestras):

        muestra = muestras[i, :]
        explicacion = np.dot(G[i].coef_, muestra) + G[i].intercept_
        prediccion = f.predict(muestra.reshape(1, -1))
        prediccion_explicacion = 1 if explicacion > 0.5 else 0

        if prediccion_explicacion != prediccion:

            error_explicacion = error_explicacion + 1

        if prediccion != test[i]:

            error_prediccion = error_prediccion + 1

    return error_explicacion / error_prediccion


def congruencia(muestras):
    n=len(coherencia)
    promedio=sum(coherencia)/n
    suma=sum((a-promedio)**2 for a in coherencia)
    result = math.sqrt(suma/n)
    return result

#medida de identidad

identidad(test_attributes[5,:], test_attributes[5,:], randomForestModel, max_attributes_adults, min_attributes_adults)

#medida de separabilidad

print(separabilidad(test_attributes[5,:], test_attributes[8,:], randomForestModel, max_attributes_adults, min_attributes_adults))
'''
#medida de selectividad
attribute_training_modificado = training_attributes.copy()

filas, columnas = attribute_training_modificado.shape


for j in range(columnas):

    d = {}

    for i in range(filas):

        if attribute_training_modificado[i, j] in d:

            d[attribute_training_modificado[i, j]] = d[attribute_training_modificado[i, j]] + 1

        else:

            d[attribute_training_modificado[i, j]] = 1

    valor_top_1 = sorted(list(d.values()), reverse=False)[:1]

    print(valor_top_1)


nuevo_orden = [11, 10, 13, 8, 1, 9, 12, 5, 7, 3, 4, 6, 0, 2]

print(selectividad(test_attributes, test_goal, randomForestModel, nuevo_orden))

#medida de coherencia
diccionario_variables_irrelevantes = {}

attribute_test_modificado = test_attributes.copy()

for i in range(filas):

    for j in range(columnas):

        if attribute_training_modificado[i, j] in diccionario_variables_irrelevantes:

            diccionario_variables_irrelevantes[attribute_training_modificado[i, j]] = diccionario_variables_irrelevantes[attribute_training_modificado[i, j]] + 1

        else:

            diccionario_variables_irrelevantes[attribute_training_modificado[i, j]] = 1

claves_valor_1 = [clave for clave, valor in diccionario_variables_irrelevantes.items() if valor == 1]

filas, columnas = attribute_test_modificado.shape

for i in range(filas):

    for j in range(columnas):

        if attribute_test_modificado[i, j] in claves_valor_1:

            attribute_test_modificado[i, j] = 0

predicciones = [xgboostModel.predict(xgb.DMatrix(np.array(x).reshape(1, -1))) for x in test_attributes]

errores_prediccion = []
errores_prediccion_modificada = []
predicciones_modificadas = []

for i in range(len(predicciones)):

    if predicciones[i] == test_goal[i]:

        error_pred = 0

    else:

        error_pred = 1
    
    errores_prediccion.append(error_pred)

num_features = training_attributes.shape[1]

for i in range(len(predicciones)):

    y_pred_modified = xgboostModel.predict(xgb.DMatrix(np.array(attribute_test_modificado[i][:]).reshape(1, -1)))

    predicciones_modificadas.append(y_pred_modified)

for i in range(len(predicciones_modificadas)):

    if predicciones_modificadas[1] == test_goal[1]:

        error_pred_mod = 0

    else:

        error_pred_mod = 1
    
    errores_prediccion_modificada.append(error_pred_mod)

for i in range(len(errores_prediccion)):

    print(coherencia(errores_prediccion[i], errores_prediccion_modificada[i]))


'''
#medida completitud

muestras = test_attributes[5:100,:]
test = test_goal[5:100]
lista_G = []
for x in muestras:
    a, b, c = LIMEAlgorithm(x,randomForestModel, 100, max_attributes_adults,min_attributes_adults)
    lista_G.append(c)

print(completitud(muestras, lista_G, randomForestModel, test))

#medida de congruencia

muestras = test_attributes[5:100]

