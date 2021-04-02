import numpy as np

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
    
    Gfx[n-1] = 200.0 * (x[n-1] - x[n-2]* x[n-2])
    
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
        Hf[i][i]   = 1200.0 * x[i] * x[i] - 400.0 * x[i+1] + 2.0
        Hf[i][i+1] = - 400.0 * x[i]
    
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

def f3 (x, params = {}) :
    '''
    Calcula el valor de f(x) del ejercicio 3
    Parámetros
    -----------
        x      : vector de valores iniciales [x1, x2, ..., xn]
        lambda : Pronderacón x_{i+1} y x_i
        sigma  : varianza de N(0, sigma)
    Regresa
        Valor f(x, lambda, sigma)
    -----------
    '''
    n  = x.shape[0]
    lam = params['lambda']
    eta = params['eta']
    sigma = params['sigma']
    
    fx_1 = 0.0
    fx_2 = 0.0

    for i in range(n) :
        t_i   = 2.0 * i / (n - 1.0) - 1.0
        y_i   = t_i * t_i + eta
        fx_1 += (x[i] - y_i) * (x[i] - y_i)
    
    for i in range(n-1) :
        fx_2 += (x[i+1] - x[i]) * (x[i+1] - x[i])

    return fx_1 + lam * fx_2

def f3_Grad (x, params = {}) :
    '''
    Calcula el gradiente de la función f3
    Parámetros
    -----------
         x  : Array de valores [x_1, x_2, ..., x_n]
    Regresa
    -----------
        Gfx : Vector gradiente Gfx
    '''
    lam = params['lambda']
    eta = params['eta']
    sigma = params['sigma']
    
    n   = x.shape[0]
    Gfx = np.zeros(n, dtype = np.float64)

    # Primera entrada
    y_0    = 1.0 + eta
    Gfx[0] = 2.0 * (x[0] - y_0) - 2.0 * lam * (x[1] - x[0])

    rango = [i+1 for i in range(n-2)]
    for i in rango :
        t_i    = 2.0 * i / (n - 1.0) - 1
        y_i    = t_i * t_i + eta
        Gfx[i] = 2.0 * (x[i] - y_i) - 2.0 * lam * (x[i+1] - x[i]) + 2.0 * lam * (x[i] - x[i-1])

    y_n     = 1.0 + eta
    Gfx[n-1] = 2.0 * (x[n-1] - y_n) - 2.0 * lam * (x[n-1] - x[n-2])

    return Gfx

