################ Identificar líneas en vías para automóviles #################

#Felipe Rojas

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import cv2 ## OPEN CV library



def escala_grises(img):
    #Transformar la image a escala de grises
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def suavizado_gaussiano(img, tamaño_mascara=5):
    #Remover "ruidos" y no linearidades
    return cv2.GaussianBlur(img, (tamaño_mascara, tamaño_mascara), 0)

def canny(img, lim_inf, lim_sup):
    #Usamos el algoritmo de Canny para encontrar bordes
    return cv2.Canny(img, lim_inf, lim_sup)

def area_de_interes(img, vertices):
    # Crear y aplicar una mascara a la imagen para mostrar SOLO lo que nos interesa ver
    mascara = np.zeros_like(img)
    
    if len(img.shape)>2:
        channel_count = img.shape[2]
        ignore_mask_color = (255, ) * channel_count
        
    else:
        ignore_mask_color = 255
    
    #Llenar los pixeles dentro del poligono deseado
    cv2.fillPoly(mascara, vertices, ignore_mask_color)
    plt.imshow(mascara, cmap = 'gray')
    
    imagen_con_mascara = cv2.bitwise_and(img, mascara)
    return imagen_con_mascara

def lineas_hough(img, rho, theta, limite, lon_min_linea, espacio_max_lineas):
    #Se necesita una imagen transformada en Canny
    
    lineas = cv2.HoughLinesP(img, rho, theta, limite, np.array([]), minLineLength = lon_min_linea, 
                            maxLineGap = espacio_max_lineas)
    img_lineas = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
    dibujar_carriles(img_lineas, lineas)
    return img_lineas


def dibujar_lineas(img, lineas, color=[255, 0,0], espesor = 10):
    for linea in lineas:
        for x1,y1,x2,y2 in linea:
            cv2.line(img,(x1,y1),(x2,y2),color, espesor)
       
            
       
def dibujar_carriles(img, lineas, color=[0, 0,255], espesor = 12):
    
    m = 0
    b = 0
    #Variables para el carril izquierdo
    m_izq_acum = 0
    b_izq_acum = 0
    cont_izq = 0
    global ult_m_izq
    global ult_b_izq
    #Variables para el carril derecho
    m_der_acum = 0
    b_der_acum = 0
    cont_der = 0
    global ult_m_der
    global ult_b_der
    
    for linea in lineas:
        for x1,y1,x2,y2 in linea:
            #Calculamos la pendiente para saber si es carril derecho o izquierdo
            #pendiente positiva es carril derecho y pendiente negativa es carril izquierdo
            m = (y2-y1)/(x2-x1)
            b = y2 - (m*x2)
            
            if m<0:
                m_izq_acum += m
                b_izq_acum += b
                cont_izq += 1
                       
            else:
                m_der_acum += m
                b_der_acum += b
                cont_der += 1
                
    #Calculamos las pendientes promedio y los puntos para cada carril
    #CARRIL IZQUIERDO
    if cont_izq != 0:
        m_izq_prom = m_izq_acum/cont_izq
        b_izq_prom = b_izq_acum/cont_izq
        ult_m_izq = m_izq_prom
        ult_b_izq = b_izq_prom
    else:
        m_izq_prom = ult_m_izq
        b_izq_prom = ult_b_izq
         
    Y_abajo = 540
    Y_arriba = 337
    X_abajo = int((Y_abajo - b_izq_prom)/m_izq_prom)
    X_arriba = int((Y_arriba - b_izq_prom)/m_izq_prom)
    cv2.line(img,(X_abajo,Y_abajo),(X_arriba,Y_arriba),color, espesor)     

    #CARRIL DERECHO     
    if cont_der != 0:
        m_der_prom = m_der_acum/cont_der
        b_der_prom = b_der_acum/cont_der
        ult_m_der = m_der_prom
        ult_b_der = b_der_prom
    else:
        m_der_prom = ult_m_der
        b_der_prom = ult_b_der
         
    X_abajo = int((Y_abajo - b_der_prom)/m_der_prom)
    X_arriba = int((Y_arriba - b_der_prom)/m_der_prom)
    cv2.line(img,(X_abajo,Y_abajo),(X_arriba,Y_arriba),color, espesor) 
              

def sumar_imagenes(img, original, alpha = 0.7, beta = 1):
    #Dibujamos las lineas encontradas en la imagen original
    return cv2.addWeighted(original, alpha, img, beta, 0)


#Funcion para dibujar las lineas entrecortadas de la carretera
    
def detecion_lineas(img):
    #Creamos una copia de la imagen
    imagen_color = np.copy(img)
    #Primero, escalar la imagen a gris
    gris = escala_grises(img)
    #Luego, suavizamos la imagen
    img_suavizada = suavizado_gaussiano(gris, 7)
    #Ahora tenemos que detectar los bordes con Canny
    img_canny = canny(img_suavizada, 50, 150)
    
    #definimos los vertices de nuestra area de interes
    imshape = img.shape
    vertices = np.array([[(48, 540),    #Vertice inferior izquierda
                          (436, 337),   #Vertice superior izquierda
                          (533, 337),   #Vertice superior derecha
                          (955,540)]])  #Vertice inferior izquierda
    
    #Recortar la imagen para solo ver la parte que nos interesa 
    img_recortada = area_de_interes(img_canny, vertices)
    
    #Definimos los parametros para la funcion de Hough
    rho = 1
    theta = np.pi/180
    limite = 20
    longitud = 20
    espacio = 15
    
    #Identificar y dibujar las lineas en una imagen en blanco
    lineas = lineas_hough(img_recortada, rho, theta, limite, longitud, espacio)
    imagen_final = sumar_imagenes(lineas, imagen_color, 0.8, 1)
    plt.imshow(imagen_final)
    return imagen_final
    

################################################################################    
########### Usamos la función para dibujar líneas entrecortadas ################
################################################################################

imagen1 = mpimg.imread('test_images/solidWhiteCurve.jpg')  
image1_entrecortada = detecion_lineas(imagen1)
mpimg.imsave("test_images/solidWhiteCurve_raw.png", image1_entrecortada)

imagen2 = mpimg.imread('test_images/solidWhiteRight.jpg')  
image2_entrecortada = detecion_lineas(imagen2)
mpimg.imsave("test_images/solidWhiteRight_raw.png", image2_entrecortada)

imagen3 = mpimg.imread('test_images/solidYellowCurve.jpg') 
image3_entrecortada = detecion_lineas(imagen3)
mpimg.imsave("test_images/solidYellowCurve_raw.png", image3_entrecortada)

imagen4 = mpimg.imread('test_images/solidYellowCurve2.jpg')  
image4_entrecortada = detecion_lineas(imagen4)
mpimg.imsave("test_images/solidYellowCurve2_raw.png", image4_entrecortada)

imagen5 = mpimg.imread('test_images/solidYellowLeft.jpg')
image5_entrecortada = detecion_lineas(imagen5)
mpimg.imsave("test_images/solidYellowLeft_raw.png", image5_entrecortada)

imagen6 = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')  
image6_entrecortada = detecion_lineas(imagen6)
mpimg.imsave("test_images/whiteCarLaneSwitch_raw.png", image6_entrecortada)

#Probamos la funcion de linea entrecortada con un video clip

from moviepy.editor import VideoFileClip

white_output = 'test_videos_output/solidWhiteRight_raw.mp4'
yellow_output = 'test_videos_output/solidYellowLeft_raw.mp4'
challenge_output = 'test_videos_output/challenge_raw.mp4'

#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
clip2 = VideoFileClip("test_videos/solidYellowLeft.mp4")
clip3 = VideoFileClip("test_videos/challenge.mp4")

white_clip = clip1.fl_image(detecion_lineas) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

yellow_clip = clip2.fl_image(detecion_lineas) #NOTE: this function expects color images!!
yellow_clip.write_videofile(yellow_output, audio=False)

challenge_clip = clip3.fl_image(detecion_lineas) #NOTE: this function expects color images!!
challenge_clip.write_videofile(challenge_output, audio=False)


##################################################################################
########### Usamos la función para dibujar los carriles de la via ################
##################################################################################

imagen1 = mpimg.imread('test_images/solidWhiteCurve.jpg')  
image1_entrecortada = detecion_lineas(imagen1)
plt.imshow(imagen1)

imagen2 = mpimg.imread('test_images/solidWhiteRight.jpg')  
image2_entrecortada = detecion_lineas(imagen2)
plt.imshow(imagen2)

imagen3 = mpimg.imread('test_images/solidYellowCurve.jpg') 
image3_entrecortada = detecion_lineas(imagen3)
plt.imshow(imagen3)

imagen4 = mpimg.imread('test_images/solidYellowCurve2.jpg')  
image4_entrecortada = detecion_lineas(imagen4)
plt.imshow(imagen4)

imagen5 = mpimg.imread('test_images/solidYellowLeft.jpg')
image5_entrecortada = detecion_lineas(imagen5)
plt.imshow(imagen5)

imagen6 = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')  
image6_entrecortada = detecion_lineas(imagen6)
plt.imshow(imagen6)

#Probamos la funcion de linea entrecortada con un video clip

from moviepy.editor import VideoFileClip

white_output = 'test_videos_output/solidWhiteRight.mp4'
yellow_output = 'test_videos_output/solidYellowLeft.mp4'
challenge_output = 'test_videos_output/challenge.mp4'

#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
clip2 = VideoFileClip("test_videos/solidYellowLeft.mp4")
clip3 = VideoFileClip("test_videos/challenge.mp4")

white_clip = clip1.fl_image(detecion_lineas) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

yellow_clip = clip2.fl_image(detecion_lineas) #NOTE: this function expects color images!!
yellow_clip.write_videofile(yellow_output, audio=False)

challenge_clip = clip3.fl_image(detecion_lineas) #NOTE: this function expects color images!!
challenge_clip.write_videofile(challenge_output, audio=False)