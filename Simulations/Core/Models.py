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







def CoCulture_Independently(t,z,
        Sr,Sb,Sg1,Sg2,ST,SI0,
        Pr,Pb,Pg1,Pg2,PT,PI0,
        kl):
    SX,PX,L = z

    #print(t) 
    Sr = 10**Sr
    Pr = 10**Pr

    Sb = 10**Sb
    Pb = 10**Pb

    Sg1 = 10**Sg1
    Pg1 = 10**Pg1

    Sg2 = 10**Sg2
    Pg2 = 10**Pg2

    kl = 10**kl

    ST = 10**ST
    PT = 10**PT

    SI0 = 10**SI0
    PI0 = 10**PI0



    if t < ST:
        Srate = Sr * np.exp(-Sb *SI0* np.exp(-Sg1*t))

    else:
        Srate = Sr * np.exp(-Sb * (1 + (SI0 * np.exp(-Sg1*ST)-1) *
            np.exp(-Sg2*(t-ST))))

    dSXdt = Srate * SX - kl*SX*L

    if t < PT:
        Prate = Pr * np.exp(-Pb * PI0 * np.exp(-Pg1*t))

    else:
        Prate = Pr * np.exp(-Pb * (1 + (PI0 * np.exp(-Pg1*PT)-1) *
            np.exp(-Pg2*(t-PT))))

    dPXdt = Prate * PX

    dLdt = PX
    #dLdt =  Prate/Pr * PX


    return[dSXdt,dPXdt,dLdt]
##############################################################################











#############################
#Fits

def Logistic(t,r,K,t0,IC):
    r = 10**r   #Maximum slope
    K = 10**K   #Endstate
    IC = 10**IC #InitialCondition
    t0 = 10**t0 #Time delay

    return IC + (K-IC) / (1 + np.exp(-r*(t-t0)))#K / (1 + ((K-IC)/IC) * np.exp(-r*t))


def Gompertz(t,b,K,t0,IC):
    b = 10**b   #
    K = 10**K   #Endtstae
    IC = 10**IC #Initial Condition
    t0 = 10**t0 #Time delay

    return IC + (K-IC) * np.exp(-np.exp(-b*(t-t0)))#IC * np.exp( np.log(K/IC) * ( -np.exp(-b*t)))








def Kinetic_Monoculture(t,z,
        kc,Y,kx):

    X,C,Se = z

    kc = 10**kc
    Y = 10**Y
    kx = 10**kx


    dXdt = -kc*X*Se + (1+Y)*kx*C

    dCdt = kc*X*Se - kx*C

    dSedt = -kc*X*Se

    return [dXdt,dCdt,dSedt]
