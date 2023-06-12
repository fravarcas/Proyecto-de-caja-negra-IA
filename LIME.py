import numpy as np
import scipy as sc
import pandas as pd
import sklearn as sk
import math
from sklearn.metrics import precision_score, roc_auc_score
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn import svm
from scipy.stats import spearmanr
from tensorflow import keras
from keras.utils import to_categorical
from sklearn.preprocessing import label_binarize

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

codificator_attributes_poker = sk.preprocessing.OrdinalEncoder()
codificator_attributes_poker.fit(attributes_poker)
codified_attributes_poker = codificator_attributes_poker.transform(attributes_poker)

codificator_goal_poker = sk.preprocessing.LabelEncoder()
codified_goal_poker = codificator_goal_poker.fit_transform(goal_poker)

max_attributes_poker = [max(codified_attributes_poker[:, x]) for x in range(10)]
min_attributes_poker = [min(codified_attributes_poker[:, x]) for x in range(10)]


(training_attributes_poker, test_attributes_poker,
 training_goal_poker, test_goal_poker) = model_selection.train_test_split(codified_attributes_poker, codified_goal_poker, random_state=12345, test_size=.33, stratify=codified_goal_poker)

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

xgboostModel_adult = xgb.train(params, training_data)

#Entrenamiento de modelo redes neuronales para datos poker_hands

attributes_neural_network = attributes_poker.to_numpy()
goal_neural_network = goal_poker.to_numpy()

classes = 10
codified_goal_poker = to_categorical(goal_neural_network, classes)

(training_attributes_neural, test_attributes_neural,
 training_goal_neural, test_goal_neural) = model_selection.train_test_split(attributes_neural_network, codified_goal_poker, random_state=12345, test_size=.33, stratify=codified_goal_poker)

normalizador = keras.layers.Normalization()
normalizador.adapt(training_attributes_neural)

neural_network = keras.Sequential()
neural_network.add(keras.Input(shape=(10,)))
neural_network.add(normalizador)
neural_network.add(keras.layers.Dense(12))
neural_network.add(keras.layers.Dense(12))
neural_network.add(keras.layers.Dense(10, activation='softmax'))
neural_network.compile(optimizer='SGD', loss='categorical_crossentropy')

neural_network.fit(training_attributes_neural, training_goal_neural,
                batch_size=256, epochs=2)

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

xgboostModel_poker = xgb.train(params, training_data_poker)


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
    if f == randomForestModel or f == neural_network:
        for i in range(len(X)):

            prediction = f.predict(np.array(X[i]).reshape(1, -1))
            Y.append(prediction)

    elif f == xgboostModel_adult or f == xgboostModel_poker:
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
#recibe como parametros dos muestras a las que calcular la distancia así como los parametros necesarios para poder obtener sus explicaciones mediante LIME
def identidad(muestra_1, muestra_2, f, max_attribute,  min_attribute):

    d_1, _, _ = LIMEAlgorithm(muestra_1, f, 100, max_attribute, min_attribute)
    d_2, _, _ = LIMEAlgorithm(muestra_2, f, 100, max_attribute, min_attribute)

    distancia_muestras = cosine_distance(muestra_1, muestra_2)

    if distancia_muestras == 1.0 or 0.9999999999999999:

        distancia_explicación = cosine_distance(np.squeeze(np.asarray(d_1)), np.squeeze(np.asarray(d_2)))

        return distancia_explicación

    else:

        print('estas muestras no son identicas')
        
        
#mismos parametros que identidad
def separabilidad(muestra_1, muestra_2, f, max_attribute,  min_attribute):

    d_1, _, _ = LIMEAlgorithm(muestra_1, f, 500, max_attribute, min_attribute)
    d_2, _, _ = LIMEAlgorithm(muestra_2, f, 500, max_attribute, min_attribute)


    distacia_muestras_ab = cosine_distance(muestra_1, muestra_2)

    if distacia_muestras_ab != 1.0:
        
        distancia_explicación = cosine_distance(np.squeeze(np.asarray(d_1)), np.squeeze(np.asarray(d_2)))
    
        return distancia_explicación

    else:

        print("estas muestras son identicas por lo tanto no se puede comprobar separabilidad")
#recibe como parametros una muestra, muestras similares y los parametros necesarios para generar una explicación
def estabilidad(muestra ,muestras_similares, f, max_attribute, min_attribute):
     
    distancias_explicaciones = []
    distancias_muestras = []
    explicacion_muestra, _, _ = LIMEAlgorithm(muestra, f, 500, max_attribute, min_attribute)

    for x in muestras_similares:

        d_1, _, _ = LIMEAlgorithm(x, f, 500, max_attribute, min_attribute)
        distancia_muestra = cosine_distance(x, muestra)
        distancia_explicacion = cosine_distance(np.squeeze(np.asarray(d_1)), np.squeeze(np.asarray(explicacion_muestra)))
        distancias_explicaciones.append(distancia_explicacion)
        distancias_muestras.append(distancia_muestra)

    correlacion, _ = spearmanr(distancias_muestras, distancias_explicaciones)

    return correlacion
#recibe como parametros los atributos y objetivos a utilizar para las predicciones así como el modelo con el que se realizan las predicciones y el orden previamente calculado en el que se van a eliminar las columnas de atributos
#de la mas a la menos relevante
def selectividad(test_attribute, test_goal, f, orden):
    if f == randomForestModel:
        y_pred_orig = f.predict(test_attribute)
    else:
        test = test_attribute
        test_xgb = xgb.DMatrix(test)
        y_pred_orig = f.predict(test_xgb)

    auc_orig = roc_auc_score(test_goal, y_pred_orig)
    auc_values = []

    for i in orden:

        attribute_test_modified = test_attribute.copy()
        attribute_test_modified[:, i] = 0 
        if f == randomForestModel:
            y_pred_modified = f.predict(attribute_test_modified)
        else:
            test = attribute_test_modified
            test_xgb = xgb.DMatrix(test)
            y_pred_modified = f.predict(test_xgb)
        auc_modified = roc_auc_score(test_goal, y_pred_modified)
        auc_change = auc_orig - auc_modified
        auc_values.append(auc_change)

    selectivity_scores = auc_values
    #la función devuelve una lista con las diferentes puntuaciones de selectividad obtenidas al ir eliminando cada columna
    return selectivity_scores

#recibe como parametros el error total del conjunto de muestras y el error total del conjunto modificado eliminando las variables irrelevantes
def coherencia(p, e):
    diferencia=abs(p-e)
    return diferencia

#recibe como parametros las muestras para calcular la metrica, y una lista con los modelos de ridge generados por la explicación lime para cada una de las muestras
#tambien recibe el modelo al que calcular la completitud y los resultados que debería de dar el modelo en el parametro test
def completitud(muestras, G, f, test, tipo):
    
    numero_de_muestras = muestras.shape[0]
    error_explicacion = 0
    error_prediccion = 0

    for i in range(numero_de_muestras):

        muestra = muestras[i, :]
        explicacion = np.dot(G[i].coef_, muestra) + G[i].intercept_
        if f == randomForestModel:
            prediccion = f.predict(muestra.reshape(1, -1))
        else:
            prediccion = f.predict(xgb.DMatrix(muestra.reshape(1, -1)))

        if tipo == 'binary':
            prediccion_explicacion = 1 if explicacion > 0.5 else 0

        elif tipo == 'multiclass':
             prediccion_explicacion = np.argmax(explicacion)

        if prediccion_explicacion != prediccion:

            error_explicacion = error_explicacion + 1

        if prediccion != test[i]:

            error_prediccion = error_prediccion + 1

    return error_explicacion / error_prediccion

#recibe las muestras a aplicar la metrica y su lista de coherencias al aplicar la metrica de coherencia a dichas muestras
def congruencia(muestras, coherencias):

    n=muestras.shape[0]
    promedio=sum(coherencias)/n
    suma=sum((a-promedio)**2 for a in coherencias)
    result = math.sqrt(suma/n)

    return result

#muestras a medir
muestras_de_medida_adult = test_attributes[:256, :]
muestras_de_medida_poker = test_attributes_poker[:256, :]

objetivos_adult = test_goal[:256]
objetivos_poker = test_goal_poker[:256]

#medida de identidad
print("medida de identidad para 256 muestras de adults: ")
for x in muestras_de_medida_adult:
    #medida de identidad para random forest y datos adult
    print("medida random forest:")
    print(identidad(x, x, randomForestModel, max_attributes_adults, min_attributes_adults))

    #medida de identidad para xgboost y datos adult
    print("medida xgboost: ")
    print(identidad(x, x, xgboostModel_adult, max_attributes_adults, min_attributes_adults))

print("medida de identidad para 256 muestras de poker: ")
for x in muestras_de_medida_poker:
    #medida de identidad para xgboost y datos poker
    print("medida xgboost: ")
    print(identidad(x, x, xgboostModel_poker, max_attributes_poker, min_attributes_poker))

    #medida de identidad para red neuronal y datos poker
    #print("medida red neuronal: ")
    #print(identidad(x, x, neural_network, max_attributes_poker, min_attributes_poker))

#medida de separabilidad
print("medida de separabilidad para 256 muestras de adults: ")
for i in range(muestras_de_medida_adult.shape[0]):
    if i < muestras_de_medida_adult.shape[0] - 1:
        #medida de separabilidad para random forest y datos adult
        print("medida random forest:")
        print(separabilidad(muestras_de_medida_adult[i,:], muestras_de_medida_adult[i+1,:], randomForestModel, max_attributes_adults, min_attributes_adults))

        #medida de separabilidad para xgboost y datos adult
        print("medida xgboost: ")
        print(separabilidad(muestras_de_medida_adult[i,:], muestras_de_medida_adult[i+1,:], xgboostModel_adult, max_attributes_adults, min_attributes_adults))

print("medida de separabilidad para 256 muestras de poker: ")
for i in range(muestras_de_medida_poker.shape[0]):
    if i < muestras_de_medida_poker.shape[0] - 1:
        #medida de separabilidad para xgboost y datos poker
        print("medida xgboost: ")
        print(separabilidad(muestras_de_medida_poker[i,:], muestras_de_medida_poker[i+1,:], xgboostModel_poker, max_attributes_poker, min_attributes_poker))



#medida de selectividad
    #calculo de los atributos más relevantes de cada columna para datos adult
    #para determinar el orden se han ordenado las columnas de más a menos usando como referencia el atributo que más aparece en cada columna

attribute_training_modificado_adult = training_attributes.copy()

filas_adult, columnas_adult = attribute_training_modificado_adult.shape

    #este for sirve para calcular cuantas veces aparece el atributo que más aparece en cada columna la lista de nuevo orden se ha formado de forma manual utilizando estos datos

for j in range(columnas_adult):

    d = {}

    for i in range(filas_adult):

        if attribute_training_modificado_adult[i, j] in d:

            d[attribute_training_modificado_adult[i, j]] = d[attribute_training_modificado_adult[i, j]] + 1

        else:

            d[attribute_training_modificado_adult[i, j]] = 1

    valor_top_1 = sorted(list(d.values()), reverse=True)[:1]

    print(valor_top_1)


nuevo_orden_adult = [11, 10, 13, 8, 1, 9, 12, 5, 7, 3, 4, 6, 0, 2]

    #calculo de los atributos más relevantes de cada columna para datos poker

attribute_training_modificado_poker = training_attributes_poker.copy()

filas_poker, columnas_poker = attribute_training_modificado_poker.shape

for j in range(columnas_poker):

    d = {}

    for i in range(filas_poker):

        if attribute_training_modificado_poker[i, j] in d:

            d[attribute_training_modificado_poker[i, j]] = d[attribute_training_modificado_poker[i, j]] + 1

        else:

            d[attribute_training_modificado_poker[i, j]] = 1

    valor_top_1 = sorted(list(d.values()), reverse=True)[:1]

    print(valor_top_1)

nuevo_orden_poker = [0, 4, 8, 6, 2, 7, 9, 3, 5, 1]

    #medida de selectividad para datos adult y random forest
print("medida de selectividad para 256 muestras de adults: ")
print("medida usando random forest:")
print(selectividad(muestras_de_medida_adult, objetivos_adult, randomForestModel, nuevo_orden_adult))

    #medida de selectividad para datos adult y xgboost
print("medida usando xgboost:")
print(selectividad(muestras_de_medida_adult, objetivos_adult, xgboostModel_adult, nuevo_orden_adult))

    #medida de selectividad para datos poker y xgboost
#print("medida de selectividad para 256 muestras de poker: ")
#print("medida usando xgboost: ")
#print(selectividad(muestras_de_medida_poker, objetivos_poker, xgboostModel_poker, nuevo_orden_poker, tipo='multiclass'))


#medida de coherencia
diccionario_variables_irrelevantes_adult = {}
diccionario_variables_irrelevantes_poker = {}

muestras_de_medida_adult_modificado = muestras_de_medida_adult.copy()
muestras_de_medida_poker_modificado = muestras_de_medida_poker.copy()
    # se buscan que atributos aparecen solo 1 vez, ya que no hay atributos que aparezcan 0 veces hemos decidido coger atributos que son casi irrelevantes esto es para adult
for i in range(filas_adult):

    for j in range(columnas_adult):

        if attribute_training_modificado_adult[i, j] in diccionario_variables_irrelevantes_adult:

            diccionario_variables_irrelevantes_adult[attribute_training_modificado_adult[i, j]] = diccionario_variables_irrelevantes_adult[attribute_training_modificado_adult[i, j]] + 1

        else:

            diccionario_variables_irrelevantes_adult[attribute_training_modificado_adult[i, j]] = 1

claves_valor_1_adult = [clave for clave, valor in diccionario_variables_irrelevantes_adult.items() if valor == 1]

    #se buscan que atributos aparecen solo 1 vez, ya que no hay atributos que aparezcan 0 veces hemos decidido coger atributos que son casi irrelevantes esto es para poker
for i in range(filas_poker):

    for j in range(columnas_poker):

        if attribute_training_modificado_poker[i, j] in diccionario_variables_irrelevantes_poker:

            diccionario_variables_irrelevantes_poker[attribute_training_modificado_poker[i, j]] = diccionario_variables_irrelevantes_poker[attribute_training_modificado_poker[i, j]] + 1

        else:

            diccionario_variables_irrelevantes_poker[attribute_training_modificado_poker[i, j]] = 1

claves_valor_1_poker = [clave for clave, valor in diccionario_variables_irrelevantes_poker.items() if valor == 1]

    #se eliminan los atributos que solo aparecen una vez del conjunto de datos adult
filas, columnas = muestras_de_medida_adult_modificado.shape

for i in range(filas):

    for j in range(columnas):

        if muestras_de_medida_adult_modificado[i, j] in claves_valor_1_adult:

            muestras_de_medida_adult_modificado[i, j] = 0
    #se eliminan los atributos que solo aparecen una vez del conjunto de datos poker
filas, columnas = muestras_de_medida_poker_modificado.shape

for i in range(filas):

    for j in range(columnas):

        if muestras_de_medida_poker_modificado[i, j] in claves_valor_1_poker:

            muestras_de_medida_poker_modificado[i, j] = 0
    #se hacen las predicciones normales y se calculan los errores esto es con xgboost y datos adult
predicciones = [xgboostModel_adult.predict(xgb.DMatrix(np.array(x).reshape(1, -1))) for x in muestras_de_medida_adult]

errores_prediccion = []
errores_prediccion_modificada = []
predicciones_modificadas = []

for i in range(len(predicciones)):

    if predicciones[i] == objetivos_adult[i]:

        error_pred = 0

    else:

        error_pred = 1
    
    errores_prediccion.append(error_pred)

num_features = training_attributes.shape[1]

for i in range(len(predicciones)):

    y_pred_modified = xgboostModel_adult.predict(xgb.DMatrix(np.array(muestras_de_medida_adult_modificado[i][:]).reshape(1, -1)))

    predicciones_modificadas.append(y_pred_modified)

for i in range(len(predicciones_modificadas)):

    if predicciones_modificadas[i] == objetivos_adult[i]:

        error_pred_mod = 0

    else:

        error_pred_mod = 1
    
    errores_prediccion_modificada.append(error_pred_mod)
    #para cada prediccion se calcula la coherencia entre la lista de predicciones con atributos eliminados y la lista normal
print("medida de coherencia para 256 muestras de datos adult: ")
print("para xgboost: ")
for i in range(len(errores_prediccion)):

    print(coherencia(errores_prediccion[i], errores_prediccion_modificada[i]))

    #para adult y random forest
predicciones = [randomForestModel.predict(x.reshape(1, -1)) for x in muestras_de_medida_adult]

errores_prediccion = []
errores_prediccion_modificada = []
predicciones_modificadas = []

for i in range(len(predicciones)):

    if predicciones[i] == objetivos_adult[i]:

        error_pred = 0

    else:

        error_pred = 1
    
    errores_prediccion.append(error_pred)

num_features = training_attributes.shape[1]

for i in range(len(predicciones)):

    y_pred_modified = randomForestModel.predict(muestras_de_medida_adult_modificado[i][:].reshape(1, -1))

    predicciones_modificadas.append(y_pred_modified)

for i in range(len(predicciones_modificadas)):

    if predicciones_modificadas[i] == objetivos_adult[i]:

        error_pred_mod = 0

    else:

        error_pred_mod = 1
    
    errores_prediccion_modificada.append(error_pred_mod)
    #para cada prediccion se calcula la coherencia entre la lista de predicciones con atributos eliminados y la lista normal
print("para random forest: ")
for i in range(len(errores_prediccion)):

    print(coherencia(errores_prediccion[i], errores_prediccion_modificada[i]))

    #para poker y xgboost
predicciones = [xgboostModel_poker.predict(xgb.DMatrix(np.array(x).reshape(1, -1))) for x in muestras_de_medida_poker]

errores_prediccion = []
errores_prediccion_modificada = []
predicciones_modificadas = []

for i in range(len(predicciones)):

    if predicciones[i] == objetivos_poker[i]:

        error_pred = 0

    else:

        error_pred = 1
    
    errores_prediccion.append(error_pred)

num_features = training_attributes_poker.shape[1]

for i in range(len(predicciones)):

    y_pred_modified = xgboostModel_poker.predict(xgb.DMatrix(np.array(muestras_de_medida_poker_modificado[i][:]).reshape(1, -1)))

    predicciones_modificadas.append(y_pred_modified)

for i in range(len(predicciones_modificadas)):

    if predicciones_modificadas[i] == objetivos_poker[i]:

        error_pred_mod = 0

    else:

        error_pred_mod = 1
    
    errores_prediccion_modificada.append(error_pred_mod)
    #para cada prediccion se calcula la coherencia entre la lista de predicciones con atributos eliminados y la lista normal
print("medida de coherencia para 256 muestras de datos poker: ")
print("para xgboost: ")
for i in range(len(errores_prediccion)):

    print(coherencia(errores_prediccion[i], errores_prediccion_modificada[i]))


#medida completitud

muestras_adult = muestras_de_medida_adult
test_adult = objetivos_adult
muestras_poker = muestras_de_medida_poker
test_poker = objetivos_poker
    #Medida de completitud para datos adult y randomforest
print("medidas de completitud para 256 muestras de datos adult: ")
lista_G = []
for x in muestras_adult:

    a, b, c = LIMEAlgorithm(x,randomForestModel, 100, max_attributes_adults,min_attributes_adults)
    lista_G.append(c)

print("para random forest:")
print(completitud(muestras_adult, lista_G, randomForestModel, test_adult, 'binary'))
    #medida de completitud para datos adult y xgboost
lista_G = []
for x in muestras_adult:

    a, b, c = LIMEAlgorithm(x,xgboostModel_adult, 100, max_attributes_adults,min_attributes_adults)
    lista_G.append(c)

print("para xgboost:")
print(completitud(muestras_adult, lista_G, xgboostModel_adult, test_adult, 'binary'))

    #medida de completitud para datos poker y xgboost
print("medidas de completitud para 256 muestras de datos poker: ")
lista_G = []
for x in muestras_poker:
    a, b, c = LIMEAlgorithm(x,xgboostModel_poker, 100, max_attributes_poker,min_attributes_poker)
    lista_G.append(c)

print("para xgboost:")
print(completitud(muestras_poker, lista_G, xgboostModel_poker, test_poker, 'multiclass'))


#medida de congruencia

muestras_adult = muestras_de_medida_adult
muestras_poker = muestras_de_medida_poker
    #medida de congruencia para adult y xgboost

    # se crea una lista con las coherencias para las diferentes muestras para ello se calculan los errores necesarios para la metrica de coherencia y se van añadiendo las coherencias a la lista
predicciones = [xgboostModel_adult.predict(xgb.DMatrix(np.array(x).reshape(1, -1))) for x in muestras_adult]
coherencias = []

errores_prediccion = []
errores_prediccion_modificada = []
predicciones_modificadas = []

for i in range(len(predicciones)):

    if predicciones[i] == objetivos_adult[i]:

        error_pred = 0

    else:

        error_pred = 1
    
    errores_prediccion.append(error_pred)

for i in range(len(predicciones)):

    y_pred_modified = xgboostModel_adult.predict(xgb.DMatrix(np.array(muestras_de_medida_adult_modificado[i][:]).reshape(1, -1)))

    predicciones_modificadas.append(y_pred_modified)

for i in range(len(predicciones_modificadas)):

    if predicciones_modificadas[1] == objetivos_adult[1]:

        error_pred_mod = 0

    else:

        error_pred_mod = 1
    
    errores_prediccion_modificada.append(error_pred_mod)

for i in range(len(errores_prediccion)):

    coherencias.append(coherencia(errores_prediccion[i], errores_prediccion_modificada[i]))

print("medidas de congruencia para 256 muestras de datos adult: ")
print("para xgboost: ")
print(congruencia(muestras_adult, coherencias))

    #medida de congruencia para adult y random forest

predicciones = [randomForestModel.predict(x.reshape(1, -1)) for x in muestras_adult]
coherencias = []

errores_prediccion = []
errores_prediccion_modificada = []
predicciones_modificadas = []

for i in range(len(predicciones)):

    if predicciones[i] == objetivos_adult[i]:

        error_pred = 0

    else:

        error_pred = 1
    
    errores_prediccion.append(error_pred)

for i in range(len(predicciones)):

    y_pred_modified = randomForestModel.predict(muestras_de_medida_adult_modificado[i][:].reshape(1, -1))

    predicciones_modificadas.append(y_pred_modified)

for i in range(len(predicciones_modificadas)):

    if predicciones_modificadas[1] == objetivos_adult[1]:

        error_pred_mod = 0

    else:

        error_pred_mod = 1
    
    errores_prediccion_modificada.append(error_pred_mod)

for i in range(len(errores_prediccion)):

    coherencias.append(coherencia(errores_prediccion[i], errores_prediccion_modificada[i]))
print("para random forest: ")
print(congruencia(muestras_adult, coherencias))

    #medida de congruencia para poker y xgboost

predicciones = [xgboostModel_poker.predict(xgb.DMatrix(np.array(x).reshape(1, -1))) for x in muestras_poker]
coherencias = []

errores_prediccion = []
errores_prediccion_modificada = []
predicciones_modificadas = []

for i in range(len(predicciones)):

    if predicciones[i] == objetivos_poker[i]:

        error_pred = 0

    else:

        error_pred = 1
    
    errores_prediccion.append(error_pred)

for i in range(len(predicciones)):

    y_pred_modified = xgboostModel_poker.predict(xgb.DMatrix(np.array(muestras_de_medida_poker_modificado[i][:]).reshape(1, -1)))

    predicciones_modificadas.append(y_pred_modified)

for i in range(len(predicciones_modificadas)):

    if predicciones_modificadas[1] == objetivos_poker[1]:

        error_pred_mod = 0

    else:

        error_pred_mod = 1
    
    errores_prediccion_modificada.append(error_pred_mod)

for i in range(len(errores_prediccion)):

    coherencias.append(coherencia(errores_prediccion[i], errores_prediccion_modificada[i]))

print("medidas de congruencia para 256 muestras de datos poker: ")
print("para xgboost: ")
print(congruencia(muestras_adult, coherencias))

#medida de estabilidad

#muestra = muestras_de_medida_adult[1, :]
#muestras_similares = muestras_de_medida_adult[2:3, :]

#print(estabilidad(muestra, muestras_similares, randomForestModel, max_attributes_adults, min_attributes_adults))
