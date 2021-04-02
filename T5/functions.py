import numpy as np
import scipy
from scipy import special
import matplotlib.pyplot as plt

def get_data_cero_one(data, c1 = 0, c2 = 1, normalize = False) :
  '''
  Extrae los datos para el modelo
  Parámetros
  -----------
    data      : Conjunto de datos con etiquetas
    c1        : Clase uno
    c2        : clase dos
    normalize : Booleano para normalización por filas
  Regresa
  -----------
    x         : Matriz de vectores de entrenamiento
    y         : Etiquetas de entrenamiento
  '''
  index = np.logical_or(data[1] == c1, data[1] == c2)
  x = data[0][index]
  y = data[1][index]

  if normalize :
    x /= np.linalg.norm(x, axis = 1)[:, None]
  return x, y

def transform_data(x) :
  '''
  Agrega dimensión a los datos de entrenamiento
  Parámetros
  -----------
    x : Datos de entrenamiento
  Regresa
    w : Matrix aumentada [ [x[1], [1]], [x[2], [1]], ..., [x[n], [1]] ]
  -----------
  '''
  w = np.zeros((x.shape[0], x.shape[1] + 1), dtype=float)

  for i in range(x.shape[0]):
    w[i] = np.concatenate((x[i], [1.0]))

  return w

def pi(beta, x) :
  '''
  Crea vector de valores pi
  Parámetros
  -----------
    beta   : vector inicial 
    x      : vector de entrenamiento 
  Regresa
  -----------
    pi     : vector de valores [pi_1, pi_2, ..., pi_n]
  '''
  n = x.shape[0]
  pi_ = []
    
  return np.array(scipy.special.expit(x @ beta))

def log_likelihood(beta, params = {}) :
  '''
  Calcula log-likelihood
  Parámetros
  -----------
    beta    : vector iterativo [beta, beta_0]
    params  : 
              x       : datos de entrenamiento
              y       : etiquetas para entrenameinto
              epsilon : tolerancia para logaritmo
  Regresa
    h       : valor f(beta, beta_0)
  -----------
  '''
  x       = params['x']
  y       = params['y']
  epsilon = params['epsilon']

  pi_          = pi(beta, x)
  one_minus_pi = 1.0 - pi_

  # Cambio valor si argumento es cero
  pi_[abs(pi_) < epsilon]                   = epsilon
  one_minus_pi[abs(one_minus_pi) < epsilon] = epsilon

  h = np.sum( y * np.log(pi_) + (1.0 - y) * np.log(one_minus_pi) )

  # regreso negativo para minimizar -h(beta, beta_0)
  return -h


def log_likelihood_grad(beta, params = {}) :
  '''
  Calcula log-likelihood
  Parámetros
  -----------
    beta    : vector iterativo [beta, beta_0]
    params  : 
              x       : datos de entrenamiento
              y       : etiquetas para entrenameinto
              epsilon : tolerancia para logaritmo
  Regresa
    g       : gradiente en beta = [beta, beta_0]
  -----------
  '''
  x       = params['x']
  y       = params['y']

  pi_          = pi(beta, x)
  one_minus_pi = 1.0 - pi_

  g = np.sum((y * (one_minus_pi) * x.T  - (1.0 - y) * pi_ * x.T), axis = 1)
  
  # regreso negativo para minimizar -h(beta, beta_0)
  return -g
  
def get_graf(f_hist, g_hist) :    
  plt.plot(range(f_hist.shape[0]), f_hist) 
  # naming the x axis 
  plt.xlabel('k') 
  # naming the y axis 
  plt.ylabel('f_k') 

  # giving a title to my graph 
  plt.title('Iteration vs Fuction Value') 

  # function to show the plot 
  plt.show() 

  plt.plot(range(g_hist.shape[0]), g_hist) 
  # naming the x axis 
  plt.xlabel('k') 
  # naming the y axis 
  plt.ylabel('g_k') 

  # giving a title to my graph 
  plt.title('Iteration vs Gradient Norm') 

  # function to show the plot 
  plt.show() 