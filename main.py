#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 12:11:41 2020
@author: Paulo Romero Martinez
@email: paulo.romero.martinez@gmail.com
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

# Libreiras npl utilizadas

nltk.download('stopwords')
nltk.download('punkt')

#%%

# =============================================================================
# Opciones de analisis generales para dirigir el proceso
# =============================================================================

#Opcion para crear el dataframe con el texto del informe y su titulo.

construirDataframe = False

# Exportar el corpus de los documentos a un csv OJO puede ser muy grande.

exportarCorpus = False

# Exportar la clasificacion final de los informes a csv

exportarClasificacion = True

# Obtener las imagenes de los clusters mediante Silhouette y Calinski-Harabasz index 

silhouette = True

#Creamos el df para ir almacenando las clasificaciones de los projectos analizados
Clasificacion = pd.DataFrame()


#%%

# =============================================================================
# Importacion y limpieza de los txt de los technical Report
# =============================================================================

# Importar el dataset

def ls2(directorio): 
    return [obj.name for obj in scandir(directorio) if obj.is_file()]
path = './tmp/'
listaTXT = ls2(path)

# Añadimos al df Calsificacion los proyectos a clasificar

Clasificacion ['Proyectos']= listaTXT

# Leemos el titulo y el txt

dataset = pd.DataFrame()
corpusOriginal = pd.DataFrame()
corpus = []
# Limpiamos y borramos las stopwords del texto de txt using regex and english spopwords
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
    
    # Cargamos los datos en el dataset por proyecto por si lo utilizamos en el futuro
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

# Borramos los informes ya analizados del direcctorio temporal
# dir = './tmp/'
# for f in os.listdir(dir):
#     os.remove(os.path.join(dir, f))

# Por si se necesita extraer en csv todo el corpus analizado
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

#create dataframe con los nombres de las palabras

# =============================================================================
# BOW model using specific keywords
# =============================================================================

#Import specific quality index keywords usin json 
import json 

with open('qualityKeywords.json', 'r') as f:
    diccionario = json.load(f)

diccionario_calidad = diccionario['calidad']

CountVec = CountVectorizer(ngram_range=(1,1), # to use bigrams ngram_range=(2,2)
                           stop_words='english',
                           #min_df = 5, # minimum number of times a word must appear
                           vocabulary = diccionario_calidad 
                           )
Count_data = CountVec.fit_transform(corpus)
cv_dataframe = pd.DataFrame(Count_data.toarray(),columns=CountVec.get_feature_names())

#Si queremos utilizar este df para posteriores metodos
xx = CountVec.fit_transform(corpus).toarray()
cv_dataframe ['Proyectos'] = listaTXT


# =============================================================================
# Conteo de la frecuencia de palabras en cas de ser util
# =============================================================================
# wordfreq = {}
# for palabra in corpus:
#     tokens = nltk.word_tokenize(palabra)
#     for token in tokens:
#         if token not in wordfreq.keys():
#             wordfreq[token] = 1
#         else:
#             wordfreq[token] += 1

#%%
# =============================================================================
# TF-IDF Model
# =============================================================================

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(xx).toarray()

#%%

# =============================================================================
# =============================================================================
# # Aplicamos el modelo para un CLUSTERING k-Means
# =============================================================================
# =============================================================================


# Importar las librerías
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Método del codo para averiguar el número óptimo de clusters
# OJO que solo tengo 5 informes por ahora
wcss = []
for i in range(1, 50):
    kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,50), wcss)
plt.title("Método del codo")
plt.xlabel("Número de Clusters")
plt.ylabel("WCSS(k)")
plt.show()


# Aplicar el método de k-means para segmentar el data set
kmeans = KMeans(n_clusters = 10, init="k-means++", max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)
#añadimos la clasificacion a df con los proyectos analizados
Clasificacion['K-means'] = y_kmeans

# =============================================================================
# SILHOUETTE analysis and graph on KMeans clustering, for several groups
# =============================================================================

if silhouette is True:

    from sklearn.metrics import silhouette_samples, silhouette_score
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    
    # Select the Number of clusters for the analysis
    
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
    
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
    
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
    
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
    
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
    
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
    
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
    
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
    
        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')
    
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')
    
        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
    
        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
    
    plt.show()
    
    print ('Se han generaldo las graficas of clusters with silhouette \
    analysis on KMeans clustering')
    
else:
    print('No se ha procedido al generacion de las graficas silhouette') 
    
# =============================================================================
# Calinski-Harabasz index 
# =============================================================================

from sklearn.metrics import calinski_harabasz_score 

# kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
labels = kmeans.labels_
calinski_harabasz_score(X, labels)

# =============================================================================
# hdbscan para vis
# =============================================================================


# %%


# =============================================================================
# =============================================================================
# # Aplicamos el modelo para un CLUSTERING Hierarchical
# =============================================================================
# =============================================================================

# Utilizar el dendrograma para encontrar el número óptimo de clusters
if exportarClasificacion is True:
    import scipy.cluster.hierarchy as sch
    dendrogram = sch.dendrogram(sch.linkage(X, method = "ward"))
    plt.title("Dendrograma")
    plt.xlabel("Projectos")
    plt.ylabel("Distancia Euclídea")
    plt.show()

    # Ajustar el clustetring jerárquico a nuestro conjunto de datos
    from sklearn.cluster import AgglomerativeClustering
    hc = AgglomerativeClustering(n_clusters = 10, affinity = "euclidean", linkage = "ward")
    y_hc = hc.fit_predict(X)
    #añadimos la clasificacion a df con los proyectos analizados
    Clasificacion['hc'] = y_hc

    # IDEA crear una grafica para visualiza los grupos


    Clasificacion.to_csv('./samples/' + 'ClasificacionFinal.csv', index=False)
    print ('Se ha exportado la clasificaion de clusterin jerarquico y k-means')
else:
    print('No se ha procedido a exportacion') 
