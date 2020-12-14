#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 12:11:41 2020
@author: Paulo Romero Martinez
@email: supertropo@gmail.com
"""
## Natural Language Processing for Mining Technical Report

# Importar librerías
import pandas as pd
import re
import nltk
import os
import numpy as np

from os import scandir

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


#%%

# =============================================================================
# Opciones de analisis generales
# =============================================================================

#Opcion para crear el dataframe con el texto del informe y su titulo

construirDataframe = False

exportarCorpus = False

nltk.download('stopwords')
nltk.download('punkt')
#Creamos el df para ir almacenando las clasificaciones de los projectos analizados
Clasificacion = pd.DataFrame()


#%%

# =============================================================================
# Importacion y limpieza de los txt de los informes
# =============================================================================

# Importar el dataset
def ls2(directorio): 
    return [obj.name for obj in scandir(directorio) if obj.is_file()]

path = './tmp/'
listaTXT = ls2(path)
#creamos un df como indice de los proyectos a clasificar
Clasificacion ['Proyectos']= listaTXT

#leer el titulo y el txt
dataset = pd.DataFrame()
corpusOriginal = pd.DataFrame()
corpus = []

for element in listaTXT: 
    # Open a file: file
    file = open(path + element,mode='r') 
    # read all lines at once
    all_of_it = file.read()
    # Limpieza de texto
    #limpiamos el texto para aligerar evitando informacion que no utilizaremos
    texto = re.sub('[^a-zA-Z]', ' ', all_of_it)
    texto = texto.lower()
    texto = texto.split()
    ps = PorterStemmer()
    texto = [ps.stem(word) for word in texto if not word in set(stopwords.words('english'))]
    texto = ' '.join(texto)
    corpus.append(texto)
    
    #cargamos los datos en el dataset por proyecto por si lo utilizamos en el futuro
    if construirDataframe is True:        
        dataset['projecto'] = [element]
        dataset['informe'] = [corpus]
        #Construimos el corpus original que se utilizadara para los modelos, por lo
        #que agregaremos la informacion de cada informe de la carpeta
        corpusOriginal =  corpusOriginal.append([dataset])
        print ('Se esta creando el CorpusOriginal de :' + element)
    else:
        print ('No se ha creado el dataset')
    # close the file (parece que no es necesario)
    file.close()

# dir = './tmp/'
# for f in os.listdir(dir):
#     os.remove(os.path.join(dir, f))

# Por si se necesita extraer todo el corpus analizado
if exportarCorpus is True:
    corpusOriginal.to_csv('corpusFinal.csv', index=False)
    print ('Se ha exportado el Corpus Original con toda la informacion')
else:
    print('No se ha procedido a exportacion') 

#%%

# =============================================================================
#   Crear el Bag of Words
# =============================================================================

from sklearn.feature_extraction.text import CountVectorizer

# Limitamos el numero maximo de features.
cv = CountVectorizer(max_features = 1500)

X = cv.fit_transform(corpus).toarray()

wordfreq = {}
for palabra in corpus:
    tokens = nltk.word_tokenize(palabra)
    for token in tokens:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1
#%%

# =============================================================================
# =============================================================================
# # Aplicamos el modelo para un CLUSTERING
# =============================================================================
# =============================================================================


# =============================================================================
# # K-Means
# =============================================================================

# Importar las librerías
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Método del codo para averiguar el número óptimo de clusters
# OJO que solo tengo 5 informes por ahora
wcss = []
for i in range(1, 5):
    kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,5), wcss)
plt.title("Método del codo")
plt.xlabel("Número de Clusters")
plt.ylabel("WCSS(k)")
plt.show()


# Aplicar el método de k-means para segmentar el data set
kmeans = KMeans(n_clusters = 3, init="k-means++", max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)
#añadimos la clasificacion a df con los proyectos analizados
Clasificacion['K-means'] = y_kmeans

# IDEA crear una grafica para visualiza los grupos



# =============================================================================
# # Clustering Jerárquico
# =============================================================================

# Utilizar el dendrograma para encontrar el número óptimo de clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = "ward"))
plt.title("Dendrograma")
plt.xlabel("Projectos")
plt.ylabel("Distancia Euclídea")
plt.show()

# Ajustar el clustetring jerárquico a nuestro conjunto de datos
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 2, affinity = "euclidean", linkage = "ward")
y_hc = hc.fit_predict(X)
#añadimos la clasificacion a df con los proyectos analizados
Clasificacion['hc'] = y_hc

# IDEA crear una grafica para visualiza los grupos
