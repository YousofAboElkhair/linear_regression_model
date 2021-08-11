def L_model_forward(X, parameters):
    
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- activation value from the output (last) layer
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
    """
    caches = []
    A = X
    L = len(parameters) // 2
    
    for l in range(1, L):
        A_prev = A 
        A,cache = linear_activation_forward(A_prev ,parameters["W" +str(l)] ,parameters["b"+str(l)], activation = "relu")
        caches.append(cache)
        
    AL,cache = linear_activation_forward(A ,parameters["W" +str(L)] ,parameters["b"+str(L)], activation = "sigmoid")
    
    caches.append(cache)
    
    return AL, caches
