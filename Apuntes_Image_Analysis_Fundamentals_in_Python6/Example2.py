#---------------------------------------------------------------------------------------------------

#Ejemplo 2: Corregir iluminación no uniforme de fondo y analizar objetos de primer plano (frente)

#--------------------------------------------------------------------------------------------------

#Como mejorar una imagen en etapa previa a su análisis. En este ejemplo corregimos la iluminación no
#uniforme del fondo y convertimos la imagen a máscara binaria para poder hacer análisis de los
#objetos en primer  plano

#------------------------------------------------------------------------------------------------

#1. Importar todos los módulos y funciones

import matplotlib.pyplot as plt

import numpy as np

import scipy

from skimage.exposure import rescale_intensity, histogram

from skimage.filters import threshold_otsu

from skimage.io import imread

from skimage.measure import label, regionprops

from skimage.morphology import area_opening, opening, disk, reconstruction

from iaf.morph.watershed import separate_neighboring_objects

from skimage.color import rgb2gray

from skimage.morphology import opening, disk

from iaf.color import get_labels_cmap

	#Esta función no puede ser importada (Sale un Error)

from skimage import img_as_ubyte

#-----------------------------------------------------------------------------------------------

#2. Leer imagen y mostrarla en escala de grises

I= imread("rice.png")
plt.imshow(I,cmap= "gray", vmin=0, vmax=255)
plt.title("rice.png")
plt.axis("off")
plt.show()

#-----------------------------------------------------------------------------------------------

#3. Preprocesar la imagen para el análisis

#En la imagen el brillo es más intenso en la parte central y superior. Como paso previo al análisis
#haremos el fondo uniforme y después convertiremos la imagen a máscara binaria.

#Para hacer la iluminación del fondomás uniforme creamos una aproximación del fondo como una imagen
#separada, y después la sustraemos a la imagen original

#Como primer paso removeremos el primer plano usando aperturas morfológicas (erosión seguida de
#dilatación)

	#Para remover los objetos de la immagen, elelemento estructural debe ser medido de forma que
	#no quepa completamente dentro de un sólo objeto (grano de arroz).

	#Usamos la función disk de skimage.morphology para crear un elemento estructural con forma
	#de disco con un radio de 15

selem= disk(15)

#Ejecutamos la apertura morgológica usando la función opening de skimage.morphology

	#No obstante, antes definimos a I como la imagen en escala de grises, pues hay problemas
	#al pasarla sin esta conversión

I_gray= rgb2gray(I)

	#Ahora convertimos con la siguiente función ya que necesitamos valores tipo uint8 (0-255)

I_gray_uint8= img_as_ubyte(I_gray)

	#Sin usar esta función la prueba no funciona y la imagen se vuelve totalmente negra

background= opening(I_gray_uint8, selem) #lento para imágenes grandes!

plt.imshow(background,cmap="gray", vmin= 0, vmax= 255)
plt.title("Fondo de imagen rice.png estimado")
plt.axis("off")
plt.show()

#Ahora podemos sustraer el fondo aproximado de la imagen de la imagen original y ver los resultados
#La imagen resultante tiene un uniforme parejo, pero un poco oscuro para el análisis

I2= I_gray_uint8 - background
plt.imshow(I2, cmap="gray", vmin=0, vmax=255)
plt.title("Background-subtracted rice.png")
plt.axis("off")
plt.show()

	#Sustracción exitosa!!!!


#¿Qué pasa con el alto costo computacional? ------------------------------------

#En dichos casos (imágenes largas) se pueden usar algoritmos y optimizaciones del paquete
#iaf.process.subtract_background, que mejoran la velocidad de la substracción de fondo

#Usando la función rescale_intensity podemos incrementar el contraste de la imagen procesada
#saturando 1% de los datos en intensidades bajas y altas

p_low, p_high= np.percentile(I2, (1, 99))
I3= rescale_intensity(I2, in_range=(p_low,p_high))

#Imagen resultante:

plt.imshow(I3,cmap="gray", vmin=0, vmax=255)
plt.title("rice.png con mejora de contraste")
plt.axis("off")
plt.show()

#OJO: Este paso no es estrictamente necesario, y es descartable si necesitaramos extraer información
#de la intensidad de los pixeles

#                                          -----------------------------------


#Para crear una versión binarizada de la imagen usaremos elalgoritmo Otsu

threshold= threshold_otsu(I3)
print (f"Threshold= {threshold}")

#Threshold= 122

#Este valor puede ser usado para binarizar imágen de la siguiente forma:

bw= I3>threshold
plt.imshow(bw, cmap="gray")
plt.title("Máscara binaria con pequeños objetos debido a ruido")
plt.axis("off")
plt.show()

#En la imagen vemos que algo del ruido genero pequeños objetos que sobrevivieron al thresholding,
#además, queremos remover los objetos de los bordes que son parcialmente visibles ya que son
#cortados por los bordes

#Podemos usar apertura de área (operación morfológica) para remover estos pequeños objetos.
#Fijamos un área mínima de 50 pixeles

bw= area_opening(bw, area_threshold=50)

plt.imshow(bw, cmap="gray")
plt.title("Final binary image rice.png (with area opening")
plt.axis("off")
plt.show()



#----------------------------------------------------------------------------------------

#4. Análisis de objetos en la imagen

#Ahora que finalizamos la binarización podemos realizar análisis de los objetos presentes

#Empezamos con componentes conectados, cuyo resultado depende de contacto entre objetos, tamaño, y
#parámetro de conectividad

labels, num= label(bw, background=0, return_num= True, connectivity= 1)
print(f"Found {num} connected components. ")

#Ahora podemos plotear con color cada objeto para un rápido análisis visual

plt.imshow(labels, cmap=get_labels_cmap(), interpolation= "nearest")
plt.title("Color-coded objects")
plt.axis("off")
plt.show()

#Podemos usar la función where() de NumPy que toma como argumentos unacomparación logica, y dos
#valores de posicionamiento en la matriz de salida, el valor a poner en la posición donde la
#comparación lógica se cumple (True) y el valor a poner en la posición donde la comparación
#lógica falla (False)

	#NO ENTENDÍ ESTOOOOOOO

bw_65= np.where(labels == 65, True, False)
plt.imshow(bw_65, cmap= "gray")
plt.title("Rice grain number 65")
plt.axis("off")
plt.show()

#Podemos computar varias propiedades de cada objeto en la imagen:

props= regionprops(labels)

#Podemos extraer todas las areas en una lista:

areas= []
for prop in props:
	areas.append(prop.area)

areas= np.array(areas)

print (f"Median area = {np.median(areas)}; max area = {areas.max()}")

#Median area = 190.0; max area = 404

#El grano de arrozmás grande posee 2 veces el área de la mediana, lo que sugiere que algunos granos no
#fueron apropiadamente segmentados, y resultaron en el mismo componente (fusión objetos)

#Veamos como se ve este grano:

#Añadimos 1 al index, pues "areas" no contiene el fondo

index= np.argmax(areas) + 1
print (f"The largest object has area {areas[index - 1]} and index {index}. ")

#The largest object has area 1635 and index 16

#Extraigamoslo y mostremoslo

bw_largest= np.where(labels == index, True, False)
plt.imshow(bw_largest, cmap="gray")
plt.title("The largest object is two mmerged rice grains")
plt.axis("off")
plt.show()

#-------------------------------------------------------------------------------------------

#Segmentación con marca de agua (watershed)

#Podemos usar segmentación watershed para separar objetos fusionados.

	#Notar que la transformada watershed por defecto de la función tiende a sobre-segmentar
	#(romper objetos)

#Dado que los detalles y formas de compensar esto son más complejas, en el ejemplo usaremos un
#algoritmo más flexible implementado en iaf.morph.watershed

labels_ws, num_ws, _ = separate_neighboring_objects(bw, labels)
print (f"Found {num_ws} connected components")

#Ploteemos los nuevos labels

plt.imshow(labels_ws,cmap=get_labels_cmap(), interpolation="nearest")
plt.title("Color-coded objects after watershed")
plt.axis("off")

#Ahora midamos las propiedades de los nuevos objetos en comparación

props_ws= regionprops(labels_ws)

areas_ws= []
for prop in props_ws:
	areas_ws.append(prop.area)

areas_ws = np.array(areas_ws)

print (f"Median area = {np.median(areas_ws)}; max area = {areas_ws.max()}")

#Median area = 606.0; max area = 941.0



