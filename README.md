# Capstone Project: Cápsula Endoscópica
**Universidad:** Universitat de Barcelona (UB)
**Estudios:** Postgrado de Introducción a la Data Science y al Machine Learning

**Integrantes del equipo:**
 - Javier Sánchez Molino
 - Sergio Bravo Allué
 - Marc Bernabé Espinosa
 - Josep Fontana Castillo
## 1. Introducción y Objetivo
A partir de 44 vídeos de exploraciones endoscópicas de los que se han extraído distintos fotogramas como imágenes, clasificarlas en un total de 14 clases diferentes según el tipo de anomalía o parte del cuerpo presente en la imagen.

Los datos consisten en un total de 47.238 imágenes etiquetadas según las 14 clases detalladas a continuación:
* A) Angiectasia
* B) Blood
* C) Erosion
* D) Erythematous
* E) Foreign Bodies
* F) Ileo-cecal valve
* G) Lymphoid Hyperplasia
* H) Normal mucosa
* I) Pylorus
* J) Reduced Mucosal View
* K) Ulcer
* Ampulla of vater
* Hematin
* Polyp

En el *paper* original [Kvasir-Capsule, a video capsule endoscopy dataset](https://osf.io/gr7bn/) se descartan las 3 últimas clases debido a su escasa representación (10, 12 y 64 imágenes, respectivamente). Así pues, el problema de clasificación se centrará en las 11 clases restantes y un total de 47.161 imágenes, con la siguiente distribución:


<font color='red'> **INTRODUIR HISTOGRAMA DE LES 11 CLASSES**

    
<font color='black'> El *pylorus* es el punto de unión del estómago y el intestino delgado, y la *ileocecal valve* marca la transición entre el intestino delgado y el grueso. De las 9 clases restantes, la *normal mucosa* corresponde a imágenes en las que se ve con claridad la mucosa del intestino delgado, mientras que la clase *reduced mucosal view* corresponde a una imagen poco clara o con elementos que dificultan la visión de la mucosa. Las 7 clases restantes sí corresponden a patologías del sistema gastrointestinal.

A pesar de la naturaleza distinta que presentan las diferentes clases no se les da un tratamiento diferenciado como problema de clasificación.

En el *paper* original se utilizan dos modelos de clasificación. Ambos corresponden a redes neuronales convolucionales (CNN, por sus siglas en inglés) con arquitecturas que han demostrado tener un buen comportamiento en la clasificación de imágenes del sistema gastrointestinal a partir de colonoscopias normales (no de imágenes procedentes de una cápsula endoscópica ingerida):

* ***ResNet-152***: arquitectura ganadora de la *ImageNet Challenge* (ILSVRC) de 2015 formada por una CNN con 152 capas que presentaba como principal innovación el uso de *skip connections*, es decir, la unión del input de una capa al output de otra capa varios niveles por encima.
    <font color='red'>**AFEGIR REFERENCIA AURÉLIEN**<font color='black'>
* ***DenseNet-161***: se caracteriza por presentar conexiones densas entre las distintas capas mediante los llamados [*dense blocks*](https://paperswithcode.com/method/dense-block), en los que las capas que lo componen están todas conectadas entre sí directamente unas con otras. Para mantener la naturaleza de retroalimentación de las CNN, cada capa dentro de un *dense block* obtiene inputs adicionales de todas las capas precedentes, y pasa su propio *feature-map* a las capas siguientes.

El objetivo del presente proyecto es replicar los resultados obtenidos en la investigación original, aplicar distintas variaciones y tratamientos para determinar si mejoran los resultados, y ampliar el estudio a otros modelos y estrategias que permitan obtener una mejor clasificación de las imágenes según distintas métricas.
      
      
      
 ## 2. Metodología
### 2.1. Datos
Se ha descargado el *dataset* original del [link](https://osf.io/dv2ag/) especificado en el *paper*. Existe un fichero comprimido por clase. Una vez descomprimidos todos, la estructura del dataset consiste en 14 carpetas, una para cada clase, con las imágenes incluidas en la carpeta de la clase correspondiente. Además, también se facilita un fichero *metadata.csv* en el que se informa de la relación entre cada imagen, el vídeo del que proviene, y la clase a la que pertenece y, en el caso de tratarse de una anomalía, se especifican los 4 puntos de la imagen que definen los vértices del marco en el que se puede observar la anomalía dentro de la imagen.

La distribución por clases de los datos originales es la siguiente:

<font color='red'>***INCLOURE HISTOGRAMA DEL DATASET ORIGINAL, INCLOENT NÚMERO D'IMATGES I PERCENTATGE SOBRE EL TOTAL***

<font color='black'>Los autores dividieron los datos en dos subgrupos (*split 0* y *split 1*) e hicieron el doble ejercicio de aplicar ambos modelos tomando el primer subgrupo de datos para entrenar, y el segundo para validar, y viceversa.

El *split 0* presenta la siguiente distribución por clases:

<font color='red'>***INCLOURE HISTOGRAMA DEL SPLIT 0***

<font color='black'>El *split 1*, en cambio, presenta la siguiente distribución por clases:

<font color='red'>***INCLOURE HISTOGRAMA DEL SPLIT 1***

<font color='black'>Vemos que...<font color='red'>***COMPLETAR AMB COMENTARIS SOBRE SI LA DISTRIBUCIÓ ENTRE CLASSES ÉS SIMILAR O NO***
    
<font color='black'>Dada una clase, si comparamos las imágenes que hay en cada uno de los dos *splits*, observamos que la división no parece haberse realizado de forma aleatoria, puesto que las imágenes de un *split* dada una clase son muy similares entre sí, pero muy diferentes entre los dos *splits*. Parece que el criterio para dividir en dos el dataset ha consistido en respetar el vídeo del que proviene cada imagen, de modo que dado un vídeo, sólo existen imágenes de ese vídeo en uno de los dos *split*, pero no en los dos. Por ejemplo, si consideramos la clase *Foreign Bodies*, las primeras imágenes del *split 0* son las siguientes:
    
<img style="border-radius:40px;" src="./images/02/02_01 01 foreign_body_split0.png">

El nombre de las imágenes indica el vídeo del que proviene, en este caso, el código del vídeo es *3ada4222967f421d* para todas ellas, y se confirma además que muchas de las imágenes son muy parecidas entre sí al corresponder a momentos del vídeo consecutivos.
    
<br>
Si lo comparamos con las primeras imágenes de la misma clase, pero del *split 1*, nos encontramos con:
    
<img style="border-radius:40px;" src="./images/02/02_01 02 foreign_body_split1.png">
    
Y corresponden a otro vídeo (*8885668afb844852*). Además, se ha confirmado que no existe en este segundo grupo ninguna imagen del vídeo *3ada4222967f421d*.

    
<font color='black'>Así pues, una de las variaciones que consideraremos más adelante (ver apartado 3.1. <font color='red'>***CREAR REFERÈNCIA DINÀMICA***<font color='black'>) será realizar otras divisiones de los datos para ver si afecta de manera significativa al resultado final.
    
<font color='red'>***EXPLICAR VERSIONS AMB MENYS RESOLUCIÓ?***

<font color='black'>
    
### 2.2. Código
El [script](https://github.com/simula/kvasir-capsule) original está desarrollado en Python y utiliza ***PyTorch*** como librería principal de Deep Learning. Sin embargo, hemos preferido hacer uso de [Keras](https://keras.io/) para el presente proyecto. ***Keras*** es una API de Deep Learning de alto nivel construida sobre [TensorFlow 2](https://www.tensorflow.org/) y su extensivo uso en toda la comunidad de usuarios y desarrolladores de Deep Learning nos ha permitido disponer de un gran número de recursos y ayudas. Además, resulta idónea para introducirse en esta disciplina y poder utilizar los modelos con mejor comportamiento sin necesidad de conocimientos previos especializados ni una comprensión profunda de su implementación.


### 2.3. Entorno de ejecución
La manipulación de todas las imágenes así como el entrenamiento de los modelos ha resultado ser muy exigente en cuanto a capacidad de memoria RAM requerida, así como potencia de cálculo. Así pues, aquellos componentes del equipo sin disponibilidad de un ordenador potente con GPU han tenido que realizar las ejecuciones en Google Collab.

El resultado de las ejecuciones incluídas en este documento se han obtenido con una máquina <font color='red'>***COMPLETAR AMB LES CARACTERÍSTIQUES DE LA MÀQUINA***.

<font color='black'>

## 3. Estudios realizados
Se presentan numerosas alternativas a probar y contrastar que además pueden combinarse entre ellas de distintas formas. A continuación se enumeran y explican las que hemos decidido estudiar, y los resultados obtenidos en cada caso.

### 3.1. Splits de los datos
Además de la subdivisión en dos grupos utilizada en el *paper* original, también hemos decidido probar dos *splits* aleatorios extras, uno con el 30% de los datos para validación y el otro con el 50%.

    
<font color='red'>***EL VALIDATION_SPLIT NO TÉ EL COMPORTAMENT ESPERAT. SEMPRE AGAFA ELS ÚLTIMS!***

<font color='black'>


~~En ambos casos, la división se ha realizado de forma **estratificada**, es decir, manteniendo la proporción original de imágenes de las distintas clases entre el subgrupo de entrenamiento y el de validación.~~
    

A continuación se muestran tres gráficas correspondientes a fijar la arquitectura *ResNet-152*, y utilizar en cada caso uno de los tres *splits* diferentes considerados:

<table><tr>
<td> 
  <p align="center" style="padding: 10px">
    <img alt="Split0_1" src="./images/03/Resnet_WithAug_dataSetSplits_0_1_Train_Val.png" width="300">
    <br>
    <em style="color: grey">Splits originales 0 y 1</em>
  </p> 
</td>
<td> 
  <p align="center">
    <img alt="Split50" src="./images/03/Resnet_WithAug_dataSetRandomSplits_50Percent_Train_Val.png" width="300">
    <br>
    <em style="color: grey">Split aleatorio 50% validation</em>
  </p> 
</td>
<td> 
  <p align="center">
    <img alt="Split30" src="./images/03/Resnet_WithAug_dataSetRandomSplits_30Percent_Train_Val.png" width="300">
    <br>
    <em style="color: grey">Split aleatorio 30% validation</em>
  </p> 
</td>
</tr></table>
    
    
Observamos que con el *split* original el modelo tiene un mal comportamiento, asociado a la presencia de ***overfitting***, al presentar una gráfica de la *function loss* creciente en función de las *epochs* para el conjunto de validación. Lo que significa que el modelo ajusta correctamente los datos de entreno (gráfica *epochs-function loss* decreciente), pero no los de validación. Sorprendentemente, con los otros dos *split* realizados de forma aleatoria parece ser que el modelo sí presenta un comportamiento correcto al tener una gráfica *epochs-function loss* decreciente.
    
Sin embargo, la interpretación de estos resultados lleva a equívoco. La clave consiste en tener en cuenta lo comentado sobre los datos y los *splits* originales en el punto 2.1 <font color='red'>***CREAR REFERÈNCIA DINÀMICA***<font color='black'>: los dos subgrupos de imágenes respetan los vídeos de los que proviene cada imagen, de modo que todas las imágenes de un vídeo van a uno de los dos subgrupos. Cuando el *split* se realiza de forma aleatoria sí se están asignando imágenes de un mismo vídeo tanto al subconjunto de entrenamiento como al de validación, y tal y como se ha comentado, las imágenes de un mismo vídeo acostumbran a ser muy parecidas entre sí.
    
Por tanto, lo que ocurre con los *splits* originales es que las imágenes de validación son muy distintas a las de entrenamiento y el modelo demuestra hacer *overfitting* y no ser capaz de generalizar correctamente. Con los *splits* aleatorios las imágenes de validación son muy parecidas a las de entrenamiento, al provenir de los mismos vídeos, y el modelo parece generalizar correctamente cuando en realidad lo que hace es clasificar correctamente imágenes que básicamente son las mismas con las que ha entrenado. Así pues, **los modelos no generalizan correctamente**.

    
### 3.2. Preprocesado y Data Augmentation
Después de [cargar](https://www.tensorflow.org/tutorials/load_data/images?hl=en) todas las imágenes mediante las utilidades específicas de Keras, el **preprocesado** básico a aplicar consiste en reescalar las imágenes para que los valores de los píxels no esté comprendido en el rango usual de [0,255] sino que cubra el intervalo de valores [-1,1] o bien [0,1].

El método de **Data Augmentation** consiste en incrementar artificialmente el número de imágenes del subrgrupo de entrenamiento mediante la generación de variantes realistas de las imágenes originales. De este modo se reduce el *overfitting* del modelo y funciona como técnica de **regularización**. <font color='red'>***INCLOURE REFERÈNCIA A LA PAG. 613 DE L'AURELIEN***

<font color='black'> Las técnicas para generar nuevas imágenes consisten en aplicar de forma aleatoria giros, simetrías, modificaciones del contraste o zooms sobre determinadas zonas de la imagen original. Ésta última opción la hemos descartado por considerar que podría llevar a equívoco al modelo si justo se hace zoom sobre una zona en la que no se encuentra la anomalía correspondiente, puesto que la imagen resultante siempre se etiqueta igual que la de partida. En el *script* original tampoco aplican zooms, pero sí el resto de técnicas. Por tanto, hemos decidido aplicarlas en todas nuestras pruebas, al considerar que generalmente dará un resultado mejor o igual a no aplicarlas.

Existe la opción de realizar la *data augmentation* como parte del preprocesado, antes de enviar las imágenes al modelo. Sin embargo, también pueden incorporarse [capas iniciales extras](https://www.tensorflow.org/tutorials/images/data_augmentation?hl=en) en la CNN que se encarguen de realizar este incremento de imágenes y que sólo aplicará cuando se trate del subconjunto de entrenamiento. Nosotros hemos optado por esta segunda opción:
    
```python
    data_augmentation = keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.6),
        # tf.keras.layers.RandomZoom(0.3),
        tf.keras.layers.RandomContrast(0.5)])
```


### 3.3. Arquitecturas de modelos CNN - Transfer Learning


De las dos arquitecturas distintas de CNN que se utilizan en el *paper* original, la **DenseNet-161** no está disponible en Keras, sólo en PyTorch. Sin embargo, después de explorar las distintas opciones de arquitecturas que nos ofrece Keras, hemos decidido sustituirla por una de parecida, la *DenseNet-169*.

Así pues, hemos utilizado tres arquitecturas de CNN distintas:

1. ***ResNet-152***
    
2. ***DenseNet-169***: tiene un nivel de *accuracy* sobre ImageNet ligeramente [inferior](https://paperswithcode.com/model/densenet?variant=densenet-161) a la *DenseNet-161*, pero las consideramos suficientemente asimilables.
    
3. ***EfficientNet-B7***: la familia de redes *EfficientNet* fueron introducidas por primera vez en el paper "[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)" de 2019, y se caracterizan por presentar un equilibrio excelente entre número de parámetros y nivel de acierto en la clasificación respecto a otras redes ampliamente utilizadas, tal y como puede verse en el gráfico de más abajo. Además, la construcción de las distintas redes de la familia se basa en el modelo base *B0* que se reescala de forma uniforme para cada dimensión de la CNN (anchura, profundidad y resolución). <font color='red'> ***AFEGIR REFERÈNCIA (https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html)***

A continuación se muestra una gráfica comparativa entre distintos modelos según el número de parámetros utilizados y el nivel de *Top-1 Accuracy* sobre la base de imágenes de ImageNet:

![ModelsComparison](https://www.researchgate.net/publication/352346653/figure/fig5/AS:1033729019486210@1623471612282/Model-Size-vs-Accuracy-Comparison-EfficientNet-B0-is-the-baseline-network-developed-by.jpg)

Se ha decidido utilizar la variante *EfficientNet-B7* por presentar el mejor comportamiento de todas las redes consideradas. Además su número de parámetros es del mismo orden de magnitud que el de la *ResNet-152* (66MM vs 60MM, respectivamente) pero presenta una *Top-1 accuracy* muy superior.
    
Para los tres modelos se ha seguido una metodología de **transfer learning** estricta en la que se han mantenido los pesos originales de los modelos entrenados con la base de imágenes de ImageNet, quitándoles sólo la última capa, correspondiente al clasificador final.
    
A continuación se incluye el código que especifica la estructura de todos los modelos utilizados. Entre los tres sólo cambia internamente el `base_model` utilizado. Las capas que se han añadido son las últimas cuatro:
    
```python
   model = tf.keras.Sequential([
        tf.keras.Input(shape=(img_height, img_width, 3)),
        data_augmentation,                                   # Data augmentation
        tf.keras.layers.Rescaling(1./127.5, offset=-1),      # Data rescalation
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(num_classes)]) 
```
    
    
<font color='red'>***EXPLICAR LES CAPES EXTRES QUE HEM AFEGIT***
    
<font color='black'>
    
Si fijamos como *split* a utilitzar el original del *paper*, los resultados para los distintos modelos son:
    
<table><tr>
<td> 
  <p align="center" style="padding: 10px">
    <img alt="ResNet" src="./images/03/Resnet_WithAug_dataSetSplits_0_1_Train_Val.png" width="300">
    <br>
    <em style="color: grey">Modelo ResNet-152</em>
  </p> 
</td>
<td> 
  <p align="center">
    <img alt="DenseNet" src="./images/03/Desnet_WithAug_dataSetSplits_0_1_Train_Val.png" width="300">
    <br>
    <em style="color: grey">Modelo DenseNet-169</em>
  </p> 
</td>
<td> 
  <p align="center">
    <img alt="EfficientNet" src="./images/03/EfficientNetB7_WithAug_dataSetSplits_0_1_Train_Val.png" width="300">
    <br>
    <em style="color: grey">Modelo EfficientNet-B7</em>
  </p> 
</td>
</tr></table>


<font color='red'>***¿QUE PASSA AMB L'EFFICIENTNET-B7?***
    
<font color='black'>
    
Sí se observa cómo en el entrenamiento el modelo *ResNet-152* llega a niveles de *accuracy* inferiores respecto la *DenseNet-169*, así como a valores de la función de coste superiores.
    
<br>  
A pesar de lo comentado en el punto 3.1 <font color='red'>CREAR REFERÈNCIA DINÀMICA<font color='black'> anterior, a modo ilustrativo, realizamos la misma comparativa con el *split* aleatorio con el 30% de las imágenes en el grupo de validación:
      
<table><tr>
<td> 
  <p align="center" style="padding: 10px">
    <img alt="ResNet" src="./images/03/Resnet_WithAug_dataSetRandomSplits_30Percent_Train_Val.png" width="300">
    <br>
    <em style="color: grey">Modelo ResNet-152</em>
  </p> 
</td>
<td> 
  <p align="center">
    <img alt="DenseNet" src="./images/03/Desnet_WithAug_dataSetRandomSplits_30Percent_Train_Val.png" width="300">
    <br>
    <em style="color: grey">Modelo DenseNet-169</em>
  </p> 
</td>
<td> 
  <p align="center">
    <img alt="EfficientNet" src="./images/03/EfficientNetB7_WithAug_dataSetRandomSplits_30Percent_Train_Val.png" width="300">
    <br>
    <em style="color: grey">Modelo EfficientNet-B7</em>
  </p> 
</td>
</tr></table>


<font color='red'>***¿QUE PASSA AMB L'EFFICIENTNET-B7?***
    
<font color='black'>

Se confirma el mejor comportamiento de la *DenseNet-169* respecto la *ResNet-152*.

<br>
    
<font color='red'>***SI AL FINAL FEM PROVES REENTRENANT UNA PART DELS MODELS, EXPLICAR-HO***
    

<font color='black'>
    
    
### 3.4. Hiperparámetros de los modelos - Datos desbalanceados
#### 3.4.1. Pesos por clase

    
<font color='red'>***AQUÍ EXPLICAREM TEMA PESOS PER CLASSE I COM ELS HEM CALCULAT***
    
<font color='black'>
    
#### 3.4.2. Learning rate y Optimizador
    
<font color='red'>***INCLOURE LA TEORIA DE L'AURELIEN SOBRE EL LEARNING RATE I OPTIMITZADOR 'ADAM'***

<font color='black'>
    
### 3.5. Batch size
Tal y como se explica en el siguiente [artículo](https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network), el `batch_size` indica el número de instancias (inputs, en nuestro caso, imágenes) que se propagan a la vez a lo largo de la red neuronal. Para cada *batch* se actualizan los pesos (parámetros) de la red neuronal, porque es al nivel *batch* que se calcula la estimación del gradiente. Cuanto más pequeño sea el tamaño del *batch* menos cantidad de memoria requerirá el proceso de entrenamiento, pero peor estimación del gradiente realizará, al tener en cuenta sólo una parte de los datos en su cálculo.
    
En cambio, si todos los datos del dataset utilizados para entrenamiento caben en un único *batch*  se conseguirá la mejor estimación posible del gradiente.
    
Con la cantidad de imágenes que tenemos, debemos utilizar un tamaño de *batch* muy inferior al total de imágenes a procesar. Con el fin de determinar si puede llegar a influir el tamaño de *batch* definido, hemos realizado una prueba con un `batch_size=64`, en contraste con el tamaño habitual de `16` que hemos utilizado en el resto de ejecuciones.
    
A continuación se muestran los resultados obtenidos, con una arquitectura *ResNet-152* y los *splits* originales:
    
<table><tr>
<td> 
  <p align="center" style="padding: 10px">
    <img alt="BatchSize16" src="./images/03/Resnet_WithAug_dataSetSplits_0_1_Train_Val.png" width="300">
    <br>
    <em style="color: grey">Batch size igual a 16</em>
  </p> 
</td>
<td> 
  <p align="center">
    <img alt="BatchSize64" src="./images/03/Resnet_WithAug_dataSetSplits_0_1_BigBatc_Train_Val.png" width="300">
    <br>
    <em style="color: grey">Batch size igual a 64</em>
  </p> 
</td>
</tr></table>
    
    
Con las gráficas no se observa una diferencia sustancial, así que constrastemos las respectivas matrices de confusión:
    
<table><tr>
<td> 
  <p align="center" style="padding: 10px">
    <img alt="BatchSize16_CM" src="./images/03/Resnet_WithAug_dataSetSplits_0_1_ConfMatrix.png" width="300">
    <br>
    <em style="color: grey">Batch size igual a 16</em>
  </p> 
</td>
<td> 
  <p align="center">
    <img alt="BatchSize64_CM" src="./images/03/Resnet_WithAug_dataSetSplits_0_1_BigBatc_ConfMatrix.png" width="300">
    <br>
    <em style="color: grey">Batch size igual a 64</em>
  </p> 
</td>
</tr></table>
    
El resultado es ambiguo, puesto que en la diagonal principal de las matrices encontramos clases tanto con grados de acierto inferiores en el modelo con *batch* más pequeño (sería el comportamiento esperado), como clases (por ejemplo la 8) en las que la red con un *batch* de 64 predice peor.
    
<font color='red'>***EN REALITAT ÉS PITJOR EL MODEL AMB BATCH 64!!***

<font color='black'>
    
Lo que sí se ha confirmado es que el modelo con el mayor tamaño de *batch* se ha entrenado en un tiempo sustancialmente inferior al otro, al tener que realizar menos actualizaciones de los pesos de la red.


### 3.6. Métricas

<font color='red'>***TOT I QUE NO AFECTA AL RESULTAT, SÍ A LA MANERA D'INTERPRETAR-LO, I CREC QUE VAL LA PENA INCLOURE L'APARTAT. AQUÍ ES TRACTARÀ DE DIR QUE S'HAN UTILITZAT LES MÈTRIQUES QUE JA TENIA EL SKLEARN, SENSE NECESSITAT D'HAVER-LES HAGUT DE DEFINIR NOSALTRES MANUALMENT***

<font color='black'>
  
  
  
  
