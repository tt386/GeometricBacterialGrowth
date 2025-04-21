import numpy as np





def MonoCulture_Closed(t,z,r,b,g1,g2,T,I0):
    """
    ODE model for monoculture in a closed environment
    To be evaluated with solve_ivp

    ARGS:
    t:
        Time
    z:
        Variable, X, the number of cells
    r: float
        Maximum growth rate of the cells
    b: float
        Coefficient for the impact of inactive components
    g1: float
        Rate of component activation
    g2: float
        Rate of component inactivation
    T: float
        Time at which saturation phase begins
    I0: float
        Initial proportion of inactive components

    RETURNS:
    [dXdt]:
        the rate of change of the number of cells at time t
    
    """
    X = z

    r = 10**r
    b = 10**b
    g1 = 10**g1
    g2 = 10**g2
    T = 10**T
    I0 = 10**I0

    if t < T:
        rate = r * np.exp(-b*I0 * np.exp(-g1*t))

    else:
        rate = r * np.exp(-b * (1 + (I0*np.exp(-g1*T)-1) *
            np.exp(-g2*(t-T))))

    dXdt = rate * X

    return dXdt



def MonoCulture_Open_Saureus(t,z,r,b,g1):
    """
    ODE model for monoculture of S.aureus in an open environment
    To be evaluated with solve_ivp

    ARGS:
    t:
        Time
    z:
        Variable, X, the number of cells
    r: float
        Maximum growth rate of the cells
    b: float
        Coefficient for the impact of inactive components, capturing I0 too
    g1: float
        Rate of component activation

    RETURNS:
    [dXdt]:
        the rate of change of the number of cells at time t
    
    """
    X = z

    r = 10**r
    b = 10**b
    g1 = 10**g1

    rate = r * np.exp(-b * np.exp(-g1*t))

    dXdt = rate * X

    return dXdt


def MonoCulture_Open_Paeruginosa(t,z,r,b,g1,Ex,Ec):
    """
    ODE model for monoculture of P.aeruginosa in an open environment
    To be evaluated with solve_ivp

    ARGS:
    t:
        Time
    z:
        Variable, X, the number of cells
        Variable, Q, the concentration of quorum-sensing chemical
    r: float
        Maximum growth rate of the cells
    b: float
        Coefficient for the impact of inactive components, capturing I0 too
    g1: float
        Rate of component activation

    RETURNS:
    [dXdt,dQdt]:
        the rate of change of the number of cells and the concentration of 
        quorum-sensing chemical at time t
    
    """
    X,Q = z

    r = 10**r
    b = 10**b
    g1 = 10**g1
    Ex = 10**Ex
    Ec = 10**Ec

    rate = r * np.exp(-b * np.exp(-g1*t))

    dXdt = rate*X - Ex*X*Q * rate/r
    dQdt = rate/r * X - Ec*Q

    return [dXdt,dQdt]

