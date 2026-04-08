#-------------------------------------------------------------------------------------

#Example 1: Importación básica de imagen, procesamiento, y exportación

#-----------------------------------------------------------------------------------

#Como leer una imagen, ajustar su contraste, y "escribir" la imagen ajustada en un archivo

#-----------------------------------------------------------------------------------

#1.1. Importar todos los módulos y funciones

import matplotlib.pyplot as plt

import numpy as np

from skimage.io import imread, imsave

from skimage.exposure import histogram, rescale_intensity


#----------------------------------------

#1.2. Leer y mostrar la imagen

I= imread("pout.tif")

plt.imshow(I, cmap="gray")
plt.title("pout.tif")
plt.show()

plt.imshow(I, cmap="gray", vmin=0, vmax=255)
plt.title("pout.tif con prevención de estiramiento")
plt.show()

	#Especificamos vmin y vmax para prevenir que imshow() estire la intensidad de las imagenes de
	#forma automática

#--------------------------------------

#1.3. Inspeccionar la imagen

print (f"Image size = {I.shape}, data type= {I.dtype}, size in memory = {I.size} bytes. ")
print (f"Type of the Image variable = {type(I)}")

#-------------------------------------

# Histogramas

#La imagen parece no tener mucho contraste,pues no es ni muy oscura ni muy clara.

#Podemos crear un histograma para saber el rango de intensidad de la imagen

#IMPORTANTE: Estirar el contraste de imagen es solo aceptable para propósitos de visualización, pero
#NO PARA ALTERAR LAS INTENSIDADES ORIGINALES DE LA IMAGEN, ya que esto puede ocasionar error en los
#análisis cuantitativos

hist= histogram(I, source_range="dtype")

plt.bar(x= hist[1], height= hist[0])
plt.title("Histograma de pout.tif sin estiramiento")
plt.show()

#-------------------------------------

#1.4. Mejorar Contraste de la imagen

#Podemos mejorar el contraste de una imagen uint8 usando la función rescale_intensity de scikit-image,
#lo que hará un estiramiento de las intensidades para cubrir totalmente el rango dinámico de
#intensidades [0, 255].

	#Por defecto, la función estira las intensidades a los valores máximos y mínimos de la imagen


I_out= rescale_intensity(I)

plt.imshow(I_out, cmap="gray", vmin=100, vmax=255)
plt.title("pout.tif stretched 100, 255")
plt.show()

plt.imshow(I_out, cmap="gray", vmin=0, vmax=255)
plt.title("pot.tif stretched 0, 255")
plt.show()


#La imagen se volvió más oscura, pero el contraste no cambió mucho.

#Miremos el histograma actualizado:

hist_out= histogram(I_out, source_range="dtype")
plt.bar(hist_out[1], hist_out[0])
plt.title("Histograma pout.tif stretched 0,255")
plt.show()


#No parece verse un estiramiento significativo del histograma. En vez de estirarse para cubrir el rango
#commpleto, el histograma sólo se desplazó hacia la izquierda (pixeles son más oscuros)

#Dijimos que por default la función estira las intensidades entre los valores mínimos y máximos de la
#imágen, veamos cuales son:

mn= I.min()
mx= I.max()
print (f"Minimum image intensity is {mn}; maximum image intensity is {mx}. ")



#Minimum image intensity is 74; maximum image intensity is 224

#Con esto podemos ver que el mmínimo pudo ser estirado desde 74 hasta cero, sin embargo, sólo un número
#pequeño de pixeles luminosos cubrió todo el rango hasta 255. Estos últimos previnieron al histograma de
#un mayor estiramiento

	#Esto se visualiza  en que pixeles oscuros se oscurecieron aún más, pero pixeles luminosos
	#no cambiaron significativamente

#Para prevenir esto es común estirar un histograma considerando un porcentaje de sus intensidades
#mayores y menores como posibles outlieres

	#Típicamente consideramos el 95% de intensidades de pixeles, y descartamos el 2.5% de
	#intensidades mayores y menores

#Perooooo....


#¿Cómo sabemos que intensidades corresponden a los percentiles que necesitamos?.

	#Para esto usamosla función percentile() de NumPy:


p_low, p_high=np.percentile(I, (2.5, 97.5))
print (f"2,5th percentile is {p_low}; 97,5th percentile is {p_high}. ")

#2,5th percentile is 80.0; 97,5th percentile is 153.0.

#Ahora podemos usar estos valores como el nuevo rango límite de la función de estiramiento:

I_out_pc= rescale_intensity(I, in_range=(p_low, p_high))

#Observemos la nueva imagen e histograma:

plt.imshow(I_out_pc,cmap="gray", vmin=0, vmax=255)
plt.title("pout.tif con Histograma-Ecualizado")
plt.show()

hist_out_pc= histogram(I_out_pc, source_range="dtype")
plt.bar(hist_out_pc[1], hist_out_pc[0])
plt.title("Histogram after equalization")
plt.show()


#Ahora el histograma está bien "repartido" por todo el rango de la imagen uint8 (0, 255)

#--------------------------------------------------------------------------------------------

#1.5 Escribir Imagen Ajustada en un archivo:

#Para concluir escribimos la imagen ajustada I_out_pc a un archivo .tif usando la imagen imsave de
#skimage.io

imsave("pout_stretched.tif", I_out_pc)


