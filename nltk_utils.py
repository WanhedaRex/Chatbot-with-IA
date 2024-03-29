import numpy as np 
import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

nltk.download('punkt')

def tokenizacion(oracion):
   return nltk.word_tokenize(oracion)

def stem(palabra):
    return stemmer.stem(palabra.lower())

def bolsa_de_palabras(oracion_tokenizada, todas_las_palabras):
    oracion_tokenizada = [stem(w) for w in oracion_tokenizada]

    bolsa = np.zeros(len(todas_las_palabras), dtype=np.float32)
    for idx, w, in enumerate(todas_las_palabras):
        if w in oracion_tokenizada:
            bolsa[idx] = 1.0

    return bolsa