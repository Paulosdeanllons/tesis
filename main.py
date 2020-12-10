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

#%%

# =============================================================================
# Importacion y limpieza de los txt de los informes
# =============================================================================

# Importar el dataset
def ls2(directorio): 
    return [obj.name for obj in scandir(directorio) if obj.is_file()]

path = '/home/tropo/Code/Mining_NPLtr/samples/'
listaTXT = ls2(path)

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

#%%

# =============================================================================
# Aplicamos el modelo para un claster
# =============================================================================
