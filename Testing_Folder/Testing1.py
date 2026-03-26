#-------------------------------------------------------------------------------

#Testing 1: Testeo de lectura de archivos tif, aplicación de filtros, y segmentación


#-------------------------------------------------------------------------------

#0. Librerías y módulos a usar

from skimage.io import imread #lectura

import matplotlib.pyplot as plt #ploteo

from skimage.color import rgb2gray #conversión a grises

from scipy.ndimage import convolve #aplicar filtros

import numpy as np #crear filtros

from skimage.filters import gaussian #usar filtros gaussianos

from scipy.ndimage import gaussian_laplace #para usar filtro laplaciano-gaussiano

from skimage.filters import threshold_otsu, threshold_li, threshold_triangle

#-------------------------------------------------------------------------------

#1. Lectura del archivo:



def lector_imagen(imagen):

	"""
	Lee una imagen y la convierte en una matriz de intensidades

	Parámetros:
		imagen (str): Ruta del archivo de imagen

	Retorna:
		numpy.ndarray: Matriz de intensidades de píxeles
	"""

	img= imread(imagen)
	return img


	#print (img)  #, esto muestra el array de pixeles e intensidades de luz para una imagen dada, escaneada a partir de imread

#-------------------------------------------------------------------------------

#2. Observación de imagen

def visualizador_imagen(imagen, titulo="Imagen"):

	"""
	Muestra una imagen

	Parámetros:
		imagen (numpy.ndarray): Imagen a mostrar.

	Retorna:
		None
	"""
	plt.imshow(imagen, cmap="gray")
	plt.title(titulo)
	plt.axis("Off")
	plt.show()

#-------------------------------------------------------------------------------

#3. Conversión a escala blanco y negro

	#A pesar de lo anterior, aún no se pueden aplicar filtros a la imagen,debido a que posee los 3 canales de color. Para solucionar esto la convertiremos a blanco y negro de la siguiente forma:

def gray_scale_convertor(imagen):

	"""
	Convierte una imagen a escala de grises.

	Parámetros:
		imagen (numpy.ndarray): Imagen original

	Retorna:
		numpy.ndarray: Imagen en escala de grises
	"""

	img_gray= rgb2gray(imagen)
	return img_gray

#-----------------------------------------------------------------------------------

#4. Aplicación de filtros a la imagen

	#Ahora, aplicamos filtros a la imagen para hacer una transformación específica a los pixeles


def aplicacion_filtro_lineal_comun(imagen):

	"""
	Aplica filtros a la imagen

	Parámetros:
		Imagen (numpy.ndarray): Imagen en escala de grises


	Retorna:
		Tupla: Imagenes en escala de grises filtradas
	"""

	#definimos filtros:

	k_5x5= 1/25 *np.ones ((5,5),dtype= np.float32)

	k_9x9= 1/81 *np.ones ((9,9),dtype= np.float32)

	k_20x20= 1/400 *np.ones ((20,20),dtype= np.float32)


	#Aplicamos los filtros:

	img_f5x5= convolve(imagen, k_5x5)
	img_f9x9= convolve(imagen, k_9x9)
	img_f20x20= convolve(imagen, k_20x20)
	return img_f5x5, img_f9x9, img_f20x20

#-----------------------------------------------------------------------------------

#5. Aplicación de filtros Gaussianos:

def aplicacion_filtro_gauss(imagen):

	"""
	Aplica filtro gaussiano a una imagen

	Parámetros:
		Imagen (numpy.ndarray): Imagen en escala de grises a filtrar

	Retorna:
		Tupla: Imagenes en escala de grises con filtro gaussiano
	"""

	img_fgs1= gaussian(imagen, sigma=1.0)
	img_fgs3= gaussian(imagen, sigma=3.0)
	img_fgs10= gaussian(imagen, sigma=10.0)
	img_fgs05= gaussian(imagen, sigma=0.5)

	return img_fgs1, img_fgs3, img_fgs10, img_fgs05

#-----------------------------------------------------------------------------------

#6. Aplicación de derivadas como filtros:


def aplicacion_filtro_deriv(imagen):

	"""
	Aplica filtro mediante derivadas, y suaviza en determinados ejes

	Parámetros:
		Imagen(numpy.ndarray): Imagen en escala de grises a filtrar

	Retorna:
		Tupla: Imagenes suavizadas en ejes X e Y

	"""

	#definimos kernels:

	k_sx= np.array([
		[1, 0, -1],
		[1, 0, -1],
		[1, 0, -1]
	], dtype= np.float32)


	k_sy= k_sx.T

	#Aplicamos los filtros:

	img_fsx= convolve(imagen.astype(np.float32), k_sx) #Ejes verticales preservados
	img_fsy= convolve(imagen.astype(np.float32), k_sy) #Ejes horizontales preservados

	return img_fsx, img_fsy

#-----------------------------------------------------------------------------------

#7. Aplicación de filtro laplaciano gaussiano

	#Al ser demasiado sensible a ruido es combinado con un suavizado gaussiano

def aplicacion_filtro_lapl(imagen):

	"""
	Aplica filtro laplaciano,con suavizado gaussiano, sobre la imagen

	Parámetros:
		Imagen(nummpy.ndarray): Imagen en escala de grises a filtrar

	Retorna:
		Imagen(numpy.ndarray): Imagen con bordes demarcados

	"""
	img_f_log= -1.0 * gaussian_laplace(imagen.astype(np.float32), sigma=2.0)
	return img_f_log


	#el sigma puede ser ajustado según se necesite mayor o menor suavizado

#----------------------------------------------------------------------------------

#8. Segmentación con algoritmos (Otsu, Li, y Triángulo)

def mascara_binaria(imagen):

	"""
	Segmenta elementos de la imagen

	Parámetros:
		Imagen(numpy.ndarray): Imagen en escala de grises

	Retorna:
		Tupla: máscaras binarias de segmentación (Black and White)
	"""

	#Define los thresholds a partir de la imagen dada

	th_ot= threshold_otsu(imagen)
	th_li= threshold_li(imagen)
	th_tr= threshold_triangle(imagen)

	#Genera máscaras binarias de segmentación a partir de expresiones de relacion:

	bw_ot= imagen> th_ot
	bw_li= imagen> th_li
	bw_tr= imagen> th_tr
	return bw_ot, bw_li, bw_tr
#-----------------------------------------------------------------------------------

#Testing

#-----------------------------------------------------------------------------------

if __name__ == "__main__":

	#Lectura imagen (y ploteo original)

	img= lector_imagen("Test.tif")
	visualizador_imagen(img, "Imagen original")

	#Conversión y filtros

	img_gray= gray_scale_convertor(img)
	visualizador_imagen(img_gray, "Conversión a Escala de grises")

	img_f5x5, img_f9x9, img_f20x20= aplicacion_filtro_lineal_comun(img_gray)

	visualizador_imagen(img_f5x5, "Aplicación filtro k_5x5")
	visualizador_imagen(img_f9x9, "Aplicación filtro k_9x9")
	visualizador_imagen(img_f20x20, "Aplicación filtro k_20x20")

	img_fgs1, img_fgs3, img_fgs10, img_fgs05= aplicacion_filtro_gauss(img_gray)

	visualizador_imagen(img_fgs1, "Filtro gaussiano sigma=1.0")
	visualizador_imagen(img_fgs3, "Filtro gaussiano sigma=3.0")
	visualizador_imagen(img_fgs10, "Filtro gaussiano sigma=10.0")
	visualizador_imagen(img_fgs05, "Filtro gaussiano sigma=0.5")

	img_fsx, img_fsy= aplicacion_filtro_deriv(img_gray)

	visualizador_imagen(img_fsx, "Filtro ejes verticales")
	visualizador_imagen(img_fsy, "Filtro ejes horizontales")

	img_f_log= aplicacion_filtro_lapl(img_gray)

	visualizador_imagen(img_f_log, "Filtro laplaciano-gaussiano, sigma=2.0")

	#Segmentación

	bw_ot, bw_li, bw_tr= mascara_binaria(img_gray)

	visualizador_imagen(bw_ot, "Máscara binaria algoritmo Otsu")
	visualizador_imagen(bw_li, "Máscara binaria algoritmo Li")
	visualizador_imagen(bw_tr, "Máscara binaria algoritmo Triangle")






