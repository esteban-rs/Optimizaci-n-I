import numpy as np
import matplotlib.pyplot as plt

# Función de Rosembrock
def Rosembrock(x, params = {}):
    '''
    Parámetros
    -----------
        n    : dimensión de muestra
        x    : vector de valores [x_1, x_2, ..., x_n]
    Regresa
    -----------
        fx : Evaluación en la función de Rosembrock
    '''
    n  = x.shape[0]
    fx = 0.0
    
    for i in range(n-1) :
        fx += 100.0 * (x[i+1] - x[i]* x[i]) * (x[i+1] - x[i]* x[i])
        fx += (1 - x[i])*(1 - x[i])
    
    return fx

def Rosembrock_Grad(x, params = {}) :
    '''
    Calcula el gradiente de la función de Rosembrock
    
    Parámetros
    -----------
        x   : Vector de valores [x_1, x_2, ..., x_n]
    Regresa
    -----------
        Gfx : Vector gradiente [D_x_1 f(x), D_x_2 f(x), ..., D_x_n (fx)]

    '''
    
    n   = x.shape[0]
    Gfx = np.zeros(n, dtype = np.float64)
    
    # Gfx[0]  = - 400.0 * (x[1] - x[0]* x[0])* x[0] - 2.0 * ( 1- x[0])
    
    # val = [i+1 for i in range(n-1)]
    for i in range(n-1) :
        Gfx[i] = -400.0 * x[i+1]*x[i] + 400.0 * x[i]**3 + 2*x[i] - 2.0
    
    Gfx[n-1] = 200.0 * (x[n-1] - x[n-2]**2)
    
    return Gfx

def Rosembrock_Hess(x, params = {}) :
    '''
    Calcula la matrix Hessiana de la función de Rosembrock
    
    Parámetros
    -----------
        x  : Array de valores [x_1, x_2, ..., x_n]
    Regresa
    -----------
        Hf : Matrix Hesianna
    '''
    
    n = x.shape[0]
    Hf = np.zeros((n,n), dtype = np.float64)
    
    for i in range(n-1) :
        Hf[i][i]   = 1200.0 * x[i] ** 2 - 400.0 * x[i+1] + 2.0
        Hf[i][i+1] = - 400.0 * x[i]
    Hf[n-1][n-2] = -400.0 * x[n-2]
    Hf[n-1][n-1] = 200.0
    
    return Hf
# Función de Wood

def Wood(x, params = {}) :
    '''
    Calcula el valor de la función de Wood f(x)
    Parámetros
    -----------
         x: Array de valores [x_1, x_2, ..., x_n]
    Regresa
    -----------
        fx : Valor f(x)
    '''
    # Reviso Dimensión
    if (x.shape[0] != 4):
        print('Revisar vector x')
        return
    
    fx  = 100.0 * (x[0] * x[0] - x[1]) *  (x[0] * x[0] - x[1])
    fx += (x[2]* x[2] - 1.0) * (x[2]* x[2] - 1.0)
    fx += 90.0 * (x[2]*x[2] - x[3]) * (x[2]*x[2] - x[3])
    fx += 10.1 * ( (x[1] - 1.0) * (x[1] - 1.0) + (x[3] - 1.0) * (x[3] - 1.0) )
    fx += 19.8 * (x[1] - 1.0) * (x[3] - 1.0)
    
    return fx
    
def Wood_Grad(x, params = {}) :
    '''
    Calcula el gradiente de la función de Wood
    Parámetros
    -----------
         x  : Array de valores [x_1, x_2, ..., x_n]
    Regresa
    -----------
        Gfx : Vector gradiente Gfx
    '''
    
    # Reviso Dimensión
    if (x.shape[0] != 4):
        print('Revisar vector x')
        return
    
    n   = x.shape[0]
    Gfx = np.zeros(n, dtype = np.float64)
    
    Gfx[0] = 400.0 * (x[0]* x[0] - x[1]) * x[0] + 2.0 * (x[0] - 1.0)
    Gfx[1] = - 200.0 * (x[0]* x[0] - x[1]) + 20.2 * (x[1] - 1.0) + 19.8 * (x[3] - 1.0)
    Gfx[2] = 2.0 * (x[2] - 1.0) + 360.0 * (x[2]* x[2] - x[3]) * x[2]
    Gfx[3] = -180.0 * (x[2]* x[2] - x[3]) + 20.2 * (x[3] - 1.0) + 19.8 * (x[1] - 1)
    
    return Gfx
    
def Wood_Hess(x, params = {}) :
    '''
    Calcula matrix Hessiana de la función de Wood
    Parámetros
    -----------
         x  : Array de valores [x_1, x_2, ..., x_n]
    Regresa
    -----------
        Hfx : Matrix Hessiana Hfx
    '''
    # Reviso Dimensión
    if (x.shape[0] != 4):
        print('Revisar vector x')
        return        
    
    n   = x.shape[0]

    Hfx = np.zeros((n, n), dtype=np.float64)
    
    Hfx[0][0] = 1200.0 * x[0]*x[0] - 400.0 * x[1] + 2.0
    Hfx[0][1] = Hfx[1][0] = - 400.0 * x[0]
    Hfx[1][1] = Hfx[3][3] = 220.2
    Hfx[2][2] = 1080.0 * x[2]* x[2] - 360 * x[3] + 2.0
    Hfx[3][1] = Hfx[1][3] = 19.8
    Hfx[3][2] = Hfx[2][3] = - 360.0 * x[2]
    
    return Hfx

def Branin(x, params = {}) :
  '''
  Calcula valor de la función de Branin
  '''
  n = x.shape[0]
  a = params['a']
  b = params['b']
  c = params['c']
  r = params['r']
  s = params['s']
  t = params['t']

  f  = a * (x[1] - b * x[0]**2 + c * x[0] - r)**2
  f += s * (1.0 - t) * np.cos(x[0]) + s

  return f


def Branin_Grad (x, params = {}) :
  n = x.shape[0]
  a = params['a']
  b = params['b']
  c = params['c']
  r = params['r']
  s = params['s']
  t = params['t']

  f_grad = np.zeros(2, dtype = np.float64)

  f_grad[0]  = 2.0 * a * (x[1] - b * x[0]**2 + c * x[0] - r) * (-2.0 * b * x[0] + c)
  f_grad[0] -= s * (1.0 - t) * np.sin(x[0])

  f_grad[1] = 2.0 * a * (x[1] - b * x[0]**2 + c * x[0] - r)

  return f_grad

def Branin_Hess(x, params = {}) :
  n = x.shape[0]
  a = params['a']
  b = params['b']
  c = params['c']
  r = params['r']
  s = params['s']
  t = params['t']

  f_Hess = np.zeros((2,2), dtype = np.float64)

  f_Hess[0] = -4.0 * a * b * (x[1] - b * x[0]**2 + c * x[0] - r) 
  f_Hess[0] += 2.0 * a * (-2.0 * b * x[0] + c) **2 - s * (1.0 - t) * np.cos(x[1])

  f_Hess[0][1] = f_Hess[1][0] = 2.0 * a * ( -2.0 * b * x[0] + c)

  f_Hess[1][1] = 2.0 * a

  return f_Hess


def get_newton_direction(g_k, d_k) :
  p_k = np.linalg.solve(d_k, -g_k)   
  return p_k

def cholesky_plus_identity(A, beta = 1e-3, max_iter = 100):
  diag = np.diag(A)
  min_aii = min(diag)
  if min_aii > 0 :
    tau = 0.0
  else :
    tau = - min_aii + beta

  k = 0
  while k < max_iter :
    A = A + tau * np.eye(diag.shape[0])
    try :
      A = np.linalg.cholesky(A)
    except :
      tau = max(2.0 * tau, beta)
    else :
      k = max_iter
    k += 1

  return A

def newton_modified (params = []) :
    '''
    '''
    # Cargo parámetros
    x_k        = params['x_0']
    x_k_next   = None
    f          = params['f']
    f_grad     = params['f_grad']
    f_hess     = params['f_hess']
    max_iter   = params['max_iter']
    tau_x      = params['tau_x']
    tau_f      = params['tau_f']
    tau_f_grad = params['tau_grad']
    beta       = params['beta']
    max_iter_c = params['cholesky']['max_iter']

    # Subpámetros para la función
    if f.__name__ == 'Branin' :
      sub_params = {
                    'a' : params['a'],
                    'b' : params['b'],
                    'c' : params['c'],
                    'r' : params['r'],
                    's' : params['s'],
                    't' : params['t']
                  }
    else :
      sub_params = {}
        
    # Guardo Parámetros
    f_hist = []
    f_hist.append(f(x_k, params = sub_params))

    g_hist = []
    g_hist.append(np.linalg.norm(f_grad(x_k, params = sub_params)))
              
    # Comienza descenso
    k = 0 
    
    while True:
        # Calculo Gradiente
        g_k = f_grad(x_k, params = sub_params)
        # Calculo Hessiano
        H_k = f_hess(x_k, params = sub_params)
        # Factorizo usando Cholesky Modificado
        L = cholesky_plus_identity(H_k, beta, max_iter_c)   
        # Resuelvo sistema hacia atrás
        y   = np.linalg.solve(L, -g_k) 
        d_k = np.linalg.solve(L.T, y)

        # Calculo siguiente valor x_k+1
        x_k_next = x_k + d_k   
        
        # Guardo Parámetros
        f_hist.append(f(x_k_next, sub_params))
        g_hist.append(np.linalg.norm(f_grad(x_k_next, sub_params)))
        
                  
        # Criterios de paro
        if (k > max_iter) :
            break
            
        if np.linalg.norm(x_k_next - x_k)/max(np.linalg.norm(x_k), 1.0) < tau_x :
            break
                  
        if np.abs(f(x_k_next, sub_params) - f(x_k, sub_params)) / max(np.linalg.norm(f(x_k, sub_params)), 1.0) < tau_f :
            break
        
                  
        if np.linalg.norm(f_grad(x_k_next, sub_params)) < tau_f_grad :
            break
            
        # Guardo valor anterior   
        x_k = x_k_next       
        k   = k + 1
        
    return np.array(f_hist), np.array(g_hist), x_k_next

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

