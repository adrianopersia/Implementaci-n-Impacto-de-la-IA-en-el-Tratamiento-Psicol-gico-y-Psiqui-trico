# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Bibliotecas de procesamiento de lenguaje natural
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Bibliotecas de aprendizaje automático
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Descargar recursos de NLTK (solo la primera vez)
nltk.download('stopwords')

# 1. Cargar y explorar los datos
# Asegúrate de que el archivo 'comportamientos.csv' esté en el mismo directorio que este script
datos = pd.read_csv('comportamientos.csv')

# Mostrar las primeras filas del dataset
print("Primeras filas del dataset:")
print(datos.head())

# Verificar si hay valores nulos
print("\nValores nulos en el dataset:")
print(datos.isnull().sum())

# 2. Preprocesamiento del texto
# Definir funciones de preprocesamiento
stop_words = set(stopwords.words('spanish'))
stemmer = SnowballStemmer('spanish')

def limpiar_texto(texto):
    # Eliminar caracteres especiales y números
    texto = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s]', '', texto)
    # Convertir a minúsculas
    texto = texto.lower()
    # Tokenización
    palabras = texto.split()
    # Eliminar stop words y aplicar stemming
    palabras = [stemmer.stem(palabra) for palabra in palabras if palabra not in stop_words]
    # Unir palabras procesadas
    texto_procesado = ' '.join(palabras)
    return texto_procesado

# Aplicar el preprocesamiento al texto
datos['texto_procesado'] = datos['descripcion_comportamiento'].apply(limpiar_texto)

# Mostrar algunas descripciones originales y procesadas
print("\nEjemplos de texto original y procesado:")
for i in range(3):
    print(f"Original: {datos['descripcion_comportamiento'][i]}")
    print(f"Procesado: {datos['texto_procesado'][i]}\n")

# 3. Vectorización del texto
# Inicializar el vectorizador TF-IDF
vectorizador = TfidfVectorizer()

# Ajustar y transformar los datos
X = vectorizador.fit_transform(datos['texto_procesado'])

# Definir la variable objetivo
y = datos['patron_psicologico']

# 4. División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Entrenamiento del modelo
# Inicializar el modelo de Regresión Logística
modelo = LogisticRegression(max_iter=1000)

# Entrenar el modelo
modelo.fit(X_train, y_train)

# 6. Evaluación del modelo
# Realizar predicciones en el conjunto de prueba
y_pred = modelo.predict(X_test)

# Imprimir el reporte de clasificación
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=modelo.classes_, yticklabels=modelo.classes_)
plt.xlabel('Predicción')
plt.ylabel('Actual')
plt.title('Matriz de Confusión')
plt.show()

# 7. Ingresar una descripción de comportamiento por el usuario
comportamiento_usuario = input("\nIngresa una descripción de comportamiento: ")

# Preprocesar la entrada del usuario
comportamiento_procesado = limpiar_texto(comportamiento_usuario)

# Transformar el texto usando el vectorizador entrenado
X_usuario = vectorizador.transform([comportamiento_procesado])

# Realizar la predicción
prediccion_usuario = modelo.predict(X_usuario)

# Mostrar el resultado
print(f"\nPatrón psicológico predicho: '{prediccion_usuario[0]}'")
