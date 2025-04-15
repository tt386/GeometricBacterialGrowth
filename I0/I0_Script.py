#Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.optimize import minimize

import time

starttime = time.time()

#Data

#S mono
S_Mono = np.asarray([2550000,3400000,12750000,58666666.6666667,300000000,
    615000000,1071666666.66667,1383333333.33333,1933333333.33333,
    2133333333.33333,2750000000,2950000000,2950000000,3250000000])

#P mono
P_Mono = np.asarray([2116666.66666667,1883333.33333333,8550000,
    18666666.6666667,42166666.6666667,320000000,620000000,2083333333.33333,
    2100000000,2400000000,2350000000,2533333333.33333,2583333333.33333,
    2933333333.33333])

#S_Co
S_Co = np.asarray([3366666.66666667,4816666.66666667,12483333.3333333,
    38000000,166666666.666667,283333333.333333,325000000,478333333.333333,
    571666666.666667,630000000,251166666.666667,173333333.333333,45000000,
    27333333.3333333])

#P_Co
P_Co = np.asarray([1700000,2016666.66666667,5100000,23333333.3333333,
    130000000,320000000,666666666.666667,1783333333.33333,2266666666.66667,
    2533333333.33333,3400000000,3650000000,3400000000,3333333333.33333])


##############################################################################
##############################################################################
##############################################################################
#Models
def MonoCulture_Activity(t,z,r,b,g1,g2,T):
    X = z

    r = 10**r
    b = 10**b
    g1 = 10**g1
    g2 = 10**g2
    T = 10**T


    if t < T:
        rate = r * np.exp(-b * np.exp(-g1*t))

    else:
        rate = r * np.exp(-b * (1 + (np.exp(-g1*T)-1) * 
            np.exp(-g2*(t-T))))

    dXdt = rate * X

    return[dXdt]


def MonoCulture_Activity_I0(t,z,r,b,g1,g2,T,I0):
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

    return[dXdt]


def MonoCulture_Activity_Open(t,z,r,b,g1):
    X = z

    r = 10**r
    b = 10**b
    g1 = 10**g1

    rate = r * np.exp(-b * np.exp(-g1*t))

    dXdt = rate * X

    return dXdt

def MonoCulture_Activity_POpen(t,z,r,b,g1,Ex,Ec):
    X,C = z

    r = 10**r
    b = 10**b
    g1 = 10**g1
    Ex = 10**Ex
    Ec = 10**Ec

    rate = r * np.exp(-b * np.exp(-g1*t))

    dXdt = rate*X - Ex*X*C * rate/r
    dCdt = rate/r * X - Ec*C
    #dCdt = X - Ec*C

    return [dXdt,dCdt]

##############################################################################
def CoCulture_From_Mono(t,z,kl,ST,PT):
    SX,PX,L = z

    kl = 10**kl
    ST = 10**ST
    PT = 10**PT



    if t < ST:
        Srate = Sr * np.exp(-Sb * np.exp(-Sg1*t))

    else:
        Srate = Sr * np.exp(-Sb * (1 + (np.exp(-Sg1*ST)-1) * 
            np.exp(-Sg2*(t-ST))))



    dSXdt = Srate * SX - kl*SX*L





    if t < PT:
        Prate = Pr * np.exp(-Pb * np.exp(-Pg1*t))

    else:
        Prate = Pr * np.exp(-Pb * (1 + (np.exp(-Pg1*PT)-1) * 
            np.exp(-Pg2*(t-PT))))

    dPXdt = Prate * PX



    dLdt = PX
    #dLdt = Prate/Pr * PX

    return[dSXdt,dPXdt,dLdt]
##############################################################################

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

    #print(Sr,Pr,Sb,Pb,Sg1,Pg1,Sg2,Pg2,kl,ST,PT,SI0,PI0)


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

def Logistic(t,r,K,IC):
    r = 10**r
    K = 10**K
    IC = 10**IC

    return K / (1 + ((K-IC)/IC) * np.exp(-r*t))

    
def Gompertz(t,b,K,IC):
    b = 10**b
    K = 10**K
    IC = 10**IC
    return IC * np.exp( np.log(K/IC) * ( -np.exp(-b*t)))













def MakeObjective(Model,data,ICs):
    """
    Creates the objective function required for least squares optimisation
    Model is the function employed to model the data
    data is a list of time sequences
    ICs are the initial conditions
    """
    def objective(params):

        #Different behaviour if have a single dataset to fit
        if len(data) == 1:
            IC = params[-1]
            ICs[0] = 10**IC

            params = params[:-1]


        if len(data) == 2:
            SIC,PIC = params[len(params)-2:]

            ICs[0] = 10**SIC
            ICs[1] = 10**PIC

            params = params[:-2]

        result = solve_ivp(Model,[0,len(data[0])],ICs,args=tuple(params),
                t_eval=np.arange(len(data[0])),method='RK45')


        if len(result.y) == 1:
            output = result.y[0]
            if len(output) != len(data[0]):
                output = np.ones(len(data[0])) * 0.001
            return np.log10(output) - np.log10(data[0])

        else:
            return (np.log10(np.concatenate((result.y[0],result.y[1]))) - 
                    np.log10(np.concatenate((data[0],data[1]))))

    return objective


def MakeObjective_OffTime(Model,data,ICs,time):
    def objective(params):
        ICs = [10**params[-1]]
        
        params = params[:-1]

        result = solve_ivp(Model,[0,time[-1]],ICs,args=tuple(params),
                t_eval=time)

        return np.log10(result.y[0]) - np.log10(data[0])

    return objective


def MakeObjective_OffTimes(Model,data,ICs,times):
    def objective(params):
        IC1 = [10**params[-2]]
        
        params1 = (params[0],params[1],params[3])

        result1 = solve_ivp(Model,[0,times[0][-1]],IC1,args=tuple(params1),
                t_eval = times[0])


        IC2 = [10**params[-1]]

        params2 = (params[0],params[2],params[3])

        result2 = solve_ivp(Model,[0,times[1][-1]],IC2,args=tuple(params2),
                t_eval=times[1])

        return (np.log10(np.concatenate((result1.y[0],result2.y[0]))) - 
                np.log10(np.concatenate((data[0],data[1]))))

    return objective


def MakeObjective_POffTime(Model,data,ICs,time):

    def objective(params):

        ICs = [10**params[-1],0]

        params = params[:-1]

        result = solve_ivp(Model,[0,time[-1]],ICs,args=tuple(params),
                t_eval=time)


        output = result.y[0]
        output[output<0] = 1e-10

        return np.log10(output) - np.log10(data[0])

    return objective


def MakeObjective_Fits(Model,data):
    def objective(params):
        t = np.arange(len(data))
        result = Model(t,*params)

        return np.log10(result) - np.log10(data) 
    return objective



def Errors(result):
    # Jacobian at the solution
    J = result.jac

    # Residuals at the solution
    residuals = result.fun

    # Degrees of freedom
    n = len(residuals)
    p = len(result.x)
    dof = max(1, n - p)

    # Estimate variance of residuals (mean squared error)
    res_var = np.sum(residuals**2) / dof

    # Covariance matrix of the parameters
    cov_matrix = res_var * np.linalg.inv(J.T @ J)

    # Standard deviation (1-sigma) of each parameter
    param_std = np.sqrt(np.diag(cov_matrix))

    return param_std












#Initial Parameter Guesses
Sr=Pr = np.log10(3)             #Max growth rate
Sb=Pb = np.log10(10)            #controls location of end of lag.
Sg1=Pg1 = np.log10(1)           #uptake of energy max rate
Sg2=Pg2 = np.log10(0.1)         #Half rate of energy
ST=PT = np.log10(3)             #Time at which switching happens
kl = np.log10(10**-10)
SIC = np.log10(S_Mono[0])
PIC = np.log10(P_Mono[0])


##############################################################################

"""
##############################################
#S Monoculture
print("S. monoculture")
params_init = [Sr,Sb,Sg1,Sg2,ST,SIC]

objective = MakeObjective(MonoCulture_Activity,[S_Mono],[S_Mono[0]])
S_Mono_results = least_squares(objective,params_init,
        ftol=1e-10,f_scale=0.05,loss='soft_l1')

print(10**S_Mono_results.x)
##############################################
#P Monoculture
print("P monoculture")
params_init = [Pr,Pb,Pg1,Pg2,PT,PIC]

objective = MakeObjective(MonoCulture_Activity,[P_Mono],[P_Mono[0]])
P_Mono_results = least_squares(objective,params_init,
        ftol=1e-10,f_scale=0.05,loss='soft_l1')

print(10**P_Mono_results.x)
##############################################


Mono_S_Params =S_Mono_results.x
Mono_P_Params =P_Mono_results.x

Fitted_S_Mono = solve_ivp(MonoCulture_Activity,[0,len(S_Mono)],
        [10**Mono_S_Params[-1]],args=tuple(Mono_S_Params[:-1]),
        t_eval=np.linspace(0,len(S_Mono)-1,100))
Fitted_P_Mono = solve_ivp(MonoCulture_Activity,[0,len(P_Mono)],
        [10**Mono_P_Params[-1]],args=tuple(Mono_P_Params[:-1]),
        t_eval=np.linspace(0,len(P_Mono)-1,100))


plt.figure()
plt.semilogy(Fitted_S_Mono.t,Fitted_S_Mono.y[0])
plt.scatter(np.arange(len(S_Mono)),S_Mono)

plt.semilogy(Fitted_P_Mono.t,Fitted_P_Mono.y[0])
plt.scatter(np.arange(len(P_Mono)),P_Mono)

plt.ylim(1e5,1e10)

plt.ylabel("time")
plt.xlabel("bacteria count")


plt.savefig("Activity_Fitting.png")
plt.close()
"""

#############################################################################
#Account for I0##############################################
#S Monoculture
print("S. monoculture")
SI0 = 0
PI0 = 0


params_init = [Sr,Sb,Sg1,Sg2,ST,SI0,SIC]

lower = np.ones(len(params_init)) * -np.inf
upper = np.ones(len(params_init)) * np.inf

upper[-2] = 0

objective = MakeObjective(MonoCulture_Activity_I0,[S_Mono],[S_Mono[0]])
S_Mono_results_I0 = least_squares(objective,params_init,
        ftol=1e-10,f_scale=0.05,loss='soft_l1',
        bounds = (lower,upper))

print(10**S_Mono_results_I0.x)
Sr,Sb,Sg1,Sg2,ST,SI0,SIC = S_Mono_results_I0.x
##############################################
#P Monoculture
print("P monoculture")
params_init = [Pr,Pb,Pg1,Pg2,PT,PI0,PIC]

objective = MakeObjective(MonoCulture_Activity_I0,[P_Mono],[P_Mono[0]])
P_Mono_results_I0 = least_squares(objective,params_init,
        ftol=1e-10,f_scale=0.05,loss='soft_l1',
        bounds=(lower,upper))

print(10**P_Mono_results_I0.x)
Pr,Pb,Pg1,Pg2,PT,PI0,PIC = P_Mono_results_I0.x
##############################################


Mono_S_Params_I0 =S_Mono_results_I0.x
Mono_P_Params_I0 =P_Mono_results_I0.x

Fitted_S_Mono = solve_ivp(MonoCulture_Activity_I0,[0,len(S_Mono)],
        [10**Mono_S_Params_I0[-1]],args=tuple(Mono_S_Params_I0[:-1]),
        t_eval=np.linspace(0,len(S_Mono)-1,100))
Fitted_P_Mono = solve_ivp(MonoCulture_Activity_I0,[0,len(P_Mono)],
        [10**Mono_P_Params_I0[-1]],args=tuple(Mono_P_Params_I0[:-1]),
        t_eval=np.linspace(0,len(P_Mono)-1,100))


plt.figure()
plt.semilogy(Fitted_S_Mono.t,Fitted_S_Mono.y[0])
plt.scatter(np.arange(len(S_Mono)),S_Mono)

plt.semilogy(Fitted_P_Mono.t,Fitted_P_Mono.y[0])
plt.scatter(np.arange(len(P_Mono)),P_Mono)

plt.ylim(1e5,1e10)

plt.ylabel("time")
plt.xlabel("bacteria count")


plt.savefig("Activity_Fitting_I0.png")
plt.close()



##############################################################################
###Naive fits#################################################################
##############################################################################

FITSr=FITPr = np.log10(3)
SK = np.log10(max(S_Mono))
PK = np.log10(max(P_Mono))
SIC = np.log10(S_Mono[0])
PIC = np.log10(P_Mono[0])


########################################################
print("Fits")
print("S_Monoculture")
#Logistic
params_init = [FITSr,SK,SIC]

objective = MakeObjective_Fits(Logistic,S_Mono)
S_Mono_Logistic = least_squares(objective,params_init,
        ftol=1e-10,f_scale=0.05,loss='soft_l1')

print(10**S_Mono_Logistic.x)
Sr_L,SK_L,SIC_L = S_Mono_Logistic.x


#Gompertz
objective = MakeObjective_Fits(Gompertz,S_Mono)
S_Mono_Gompertz = least_squares(objective,params_init,
        ftol=1e-10,f_scale=0.05,loss='soft_l1')

print(10**S_Mono_Gompertz.x)
Sr_G,SK_G,SIC_G = S_Mono_Gompertz.x


#######################################################
#Comapring fits

t = np.linspace(0,len(S_Mono)-1,100)
plt.figure()
plt.semilogy(Fitted_S_Mono.t,Fitted_S_Mono.y[0],
        linewidth = 3,label='Activity')
plt.semilogy(t,Logistic(t,Sr_L,SK_L,SIC_L),
        linewidth=3,linestyle='dashed',label='Logistic Fit')
plt.semilogy(t,Gompertz(t,Sr_G,SK_G,SIC_G),
        linewidth=3,linestyle='dotted',label='Gompertz Fit')
plt.scatter(np.arange(len(S_Mono)),S_Mono,
        s=40,color='k',marker='x',zorder=10,label='Data')
plt.savefig("Mono_S_Comparisons.png")
plt.close()



#######################################################
print("P_Monoculture")
#Logistic
params_init = [FITPr,PK,PIC]

objective = MakeObjective_Fits(Logistic,P_Mono)
P_Mono_Logistic = least_squares(objective,params_init,
        ftol=1e-10,f_scale=0.05,loss='soft_l1')

print(10**P_Mono_Logistic.x)
Pr_L,PK_L,PIC_L = P_Mono_Logistic.x


#Gompertz
objective = MakeObjective_Fits(Gompertz,P_Mono)
P_Mono_Gompertz = least_squares(objective,params_init,
        ftol=1e-10,f_scale=0.05,loss='soft_l1')

print(10**P_Mono_Gompertz.x)
Pr_G,PK_G,PIC_G = P_Mono_Gompertz.x

#######################################################


#Comparing fits
t = np.linspace(0,len(P_Mono)-1,100)
plt.figure()
plt.semilogy(Fitted_P_Mono.t,Fitted_P_Mono.y[0],
        linewidth = 3,label='Activity')
plt.semilogy(t,Logistic(t,Pr_L,PK_L,PIC_L),
        linewidth=3,linestyle='dashed',label='Logistic Fit')
plt.semilogy(t,Gompertz(t,Pr_G,PK_G,PIC_G),
        linewidth=3,linestyle='dotted',label='GOompertz Fit')
plt.scatter(np.arange(len(P_Mono)),P_Mono,
        s=40,color='k',marker='x',zorder=10,label='Data')
plt.savefig("Mono_P_Comparisons.png")
plt.close()








"""
print("Coculture")
#Coculture: From the above results
Sr,Sb,Sg1,Sg2,ST,SIC =10**S_Mono_results.x 
Pr,Pb,Pg1,Pg2,PT,PIC =10**P_Mono_results.x

#New esimatimates for initical conditions
SIC = np.log10(S_Co[0])
PIC = np.log10(P_Co[0])

ST = np.log10(ST)
PT = np.log10(PT)

params_init = [kl,ST,PT,SIC,PIC]
objective = MakeObjective(CoCulture_From_Mono,[S_Co,P_Co],[SIC,PIC,0])
results = least_squares(objective,params_init,
        ftol=1e-10,f_scale=0.05,loss='soft_l1')

print(10**results.x)
kl,ST,PT,SIC,PIC = 10**results.x

ICs = [10**results.x[3],10**results.x[4],0]
Fitted = solve_ivp(CoCulture_From_Mono,[0,len(S_Co)],ICs,args=tuple(results.x[:-2]),
                t_eval=np.arange(len(S_Co)))#np.linspace(0,len(S_Co)-1,100))

plt.figure()
plt.semilogy(Fitted.t,Fitted.y[0],label='S')
plt.semilogy(Fitted.t,Fitted.y[1],label = 'P')

plt.scatter(np.arange(len(S_Co)),S_Co)
plt.scatter(np.arange(len(P_Co)),P_Co)

plt.ylim(1e5,1e10)

plt.ylabel("time")
plt.xlabel("bacteria count")


plt.savefig("Coculture_From_Mono.png")
"""
#######################################################################################

"""
print("Independent cocultuee")
#Coculture: Indep
params_init = np.log10(np.asarray([Sr,Sb,Sg1,Sg2,ST,Pr,Pb,Pg1,Pg2,PT,kl,S_Co[0],P_Co[0]]))

print(params_init)

objective = MakeObjective(CoCulture_Independently,[S_Co,P_Co],[S_Co[0],P_Co[0],0])
results = least_squares(objective,params_init,ftol=1e-10,f_scale=0.05,loss='soft_l1')


print("STD:",Errors(results))

print(10**results.x)


ICs = [10**results.x[-2],10**results.x[-1],0]
Fitted = solve_ivp(CoCulture_Independently,[0,len(S_Co)],ICs,args=tuple(results.x[:-2]),
                t_eval=np.linspace(0,len(S_Co)-1,100))

plt.figure()
plt.semilogy(Fitted.t,Fitted.y[0],label='S')
plt.semilogy(Fitted.t,Fitted.y[1],label = 'P')

plt.scatter(np.arange(len(S_Co)),S_Co)
plt.scatter(np.arange(len(P_Co)),P_Co)

plt.savefig("Coculture_Independently.png")

plt.close()
"""
######################################################################################

#FIgS2

Values = ("S_ATC_P_A14","S_ATC_P_A14_LBPLATES","S_RYC165_P_PA14","S_RYC157_P_PA14","S_RYC157_P_RYC157")

for DATA in Values:

    print("Working with DATA:",DATA)

    if DATA == "S_ATC_P_A14":
        S_Co = np.asarray([3366666.66666667,4816666.66666667,12483333.3333333,38000000,166666666.666667,283333333.333333,325000000,478333333.333333,571666666.666667,630000000,251166666.666667,173333333.333333,45000000,27333333.3333333])

        P_Co = np.asarray([1700000,2016666.66666667,5100000,23333333.3333333,130000000,320000000,666666666.666667,1783333333.33333,2266666666.66667,2533333333.33333,3400000000,3650000000,3400000000,3333333333.33333])

    elif DATA =="S_ATC_P_A14_LBPLATES":
        S_Co = np.asarray([2133333.33333333,3166666.66666667,6133333.33333333,34500000,60333333.3333333,271666666.666667,533333333.333333,900000000,413333333.333333,268333333.333333,47500000,19166666.6666667,9833333.33333333,3166666.66666667])

        P_Co = np.asarray([2350000,2416666.66666667,6033333.33333333,16333333.3333333,21500000,145000000,446666666.666667,1083333333.33333,1750000000,2000000000,2383333333.33333,2750000000,2833333333.33333,3100000000])


    elif DATA == "S_RYC157_P_RYC157":
        S_Co = np.asarray([2066666.66666667,2233333.33333333,11333333.3333333,18333333.3333333,38166666.6666667,256666666.666667,423333333.333333,1366666666.66667,2283333333.33333,2150000000,2366666666.66667,2733333333.33333,3050000000,3200000000,3783333333.33333,3983333333.33333,4800000000,4533333333.33333,3983333333.33333,3766666666.66667,3716666666.66667,2300000000,1950000000,215000000])

        P_Co = np.asarray([1316666.66666667,356666.666666667,2500000,2233333.33333333,2916666.66666667,4583333.33333333,11333333.3333333,13166666.6666667,24666666.6666667,29500000,35500000,68166666.6666667,148333333.333333,325000000,233333333.333333,265000000,341666666.666667,396666666.666667,691666666.666667,966666666.666667,670000000,1933333333.33333,1783333333.33333,1750000000])

    elif DATA == "S_RYC157_P_PA14":
        S_Co = np.asarray([3266666.66666667,3133333.33333333,5066666.66666667,13500000,34333333.3333333,270000000,318333333.333333,286666666.666667,330000000,218333333.333333,43833333.3333333,4500000,1933333.33333333,1366666.66666667])

        P_Co = np.asarray([578333.333333333,713333.333333333,2950000,4266666.66666667,37000000,285000000,858333333.333333,1083333333.33333,840000000,1106666666.66667,1833333333.33333,1700000000,2266666666.66667,2650000000])

    elif DATA == "S_RYC165_P_PA14":
        S_Co = np.asarray([2150000,2333333.33333333,3916666.66666667,21833333.3333333,40666666.6666667,236666666.666667,169666666.666667,280000000,331666666.666667,363333333.333333,233333333.333333,34666666.6666667,2650000,728333.333333333])

        P_Co = np.asarray([743333.333333333,1083333.33333333,4116666.66666667,19166666.6666667,43000000,218333333.333333,290000000,416666666.666667,603333333.333333,1200000000,1366666666.66667,1405000000,1933333333.33333,1503333333.33333])

    #Coculture: Indep
    params_init = (np.asarray([Sr,Sb,Sg1,Sg2,ST,0,Pr,Pb,Pg1,Pg2,PT,0,kl,np.log10(S_Co[0]),np.log10(P_Co[0])]))

    lowerbounds = [0,0,-2,-2,np.log10(3),-1,    0,0,-2,-2,np.log10(3),-1,   -13,    np.log10(S_Co[0])-1,np.log10(P_Co[0])-1]
    upperbounds = [np.log10(2),3,1,0,1.1,0,       np.log10(2),3,1,0,1.1,0,      -8,      np.log10(S_Co[0])+1,np.log10(P_Co[0])+1]

    if DATA == "S_RYC157_P_RYC157":
        upperbounds[10] = 2

    print(params_init)
    print(lowerbounds)
    print(upperbounds)

    objective = MakeObjective(CoCulture_Independently,[S_Co,P_Co],[S_Co[0],P_Co[0],0])
    results = least_squares(objective,params_init,ftol=1e-10,f_scale=0.05,loss='soft_l1',bounds=(lowerbounds,upperbounds))



    print(10**results.x)


    ICs = [10**results.x[-2],10**results.x[-1],0]
    Fitted = solve_ivp(CoCulture_Independently,[0,len(S_Co)],ICs,args=tuple(results.x[:-2]),
                    t_eval=np.arange(len(S_Co)))#np.linspace(0,len(S_Co)-1,100))


    
    print("S.aureus", DATA)
    print(*Fitted.y[0],sep=', ')

    print("Psuedo",DATA)
    print(*Fitted.y[1],sep=', ')


    plt.figure()
    plt.semilogy(Fitted.t,Fitted.y[0],label='S')
    plt.semilogy(Fitted.t,Fitted.y[1],label = 'P')

    plt.scatter(np.arange(len(S_Co)),S_Co)
    plt.scatter(np.arange(len(P_Co)),P_Co)


    plt.ylim(1e5,1e10)

    plt.title(str(DATA))
    plt.ylabel("time")
    plt.xlabel("bacteria count")

    plt.savefig("Coculture_Independently_" +str(DATA) + ".png")
    plt.close()




##############################################################################

#Open System Fits

S_Mono_Open = [13.5,16,19.5,20,23.5,26,27,32,29.5,40.5,43.5,57.5,76,88.5,101,135,148,246.5,394.5]

S_Mono_Open_Time = [0,0.2,0.6,0.8,1,1.2,1.4,1.7,1.9,2.2,2.3,2.6,2.8,3,3.1,3.3,3.5,3.9,4.3]

S_Co_Open = [11,11,10,10.5,10.5,10,10,10,10,9.5,10.5,10,10,11,11,11,12,13.5,14.5,22.5,27.5,43,58.5,74.5,111,235.5,433]

S_Co_Open_Time = [0,0.4,0.7,1,1.3,1.5,2,2.3,2.8,3.1,3.4,3.5,3.8,4.1,4.5,4.9,5.2,5.6,5.9,6.3,6.6,7,7.4,7.7,8.1,8.8,9.4]

print("OPEN SYSTEM")


print("S_Mono")
params_init = (Sr,Sb,Sg1,np.log10(S_Mono_Open[0]))
objective = MakeObjective_OffTime(MonoCulture_Activity_Open,[S_Mono_Open],
        [S_Mono_Open[0]],S_Mono_Open_Time)
results = least_squares(objective,params_init,ftol=1e-10,f_scale=0.05,loss='soft_l1')

print(10**results.x)

Sr,Sb,Sg1,S0 = results.x
ICs = [10**S0]
params = (Sr,Sb,Sg1)

Fitted = solve_ivp(MonoCulture_Activity_Open,[0,S_Mono_Open_Time[-1]],ICs,args=tuple(params),
                t_eval=S_Mono_Open_Time)

plt.figure()
plt.semilogy(Fitted.t,Fitted.y[0])
plt.scatter(S_Mono_Open_Time,S_Mono_Open)

plt.savefig("OpenSystem_SMono.png")
plt.close()
#############################################################################
print("S_Co")

params_init = (Sr,Sb,Sg1,np.log10(S_Co_Open[0]))
objective = MakeObjective_OffTime(MonoCulture_Activity_Open,[S_Co_Open],
        [S_Co_Open[0]],S_Co_Open_Time)
results = least_squares(objective,params_init,ftol=1e-10,f_scale=0.05,loss='soft_l1')

print(10**results.x)

Sr,Sb,Sg1,S0 = results.x
ICs = [10**S0]
params = (Sr,Sb,Sg1)

Fitted = solve_ivp(MonoCulture_Activity_Open,[0,S_Co_Open_Time[-1]],ICs,args=tuple(params),
                t_eval=S_Co_Open_Time)

plt.figure()
plt.semilogy(Fitted.t,Fitted.y[0])
plt.scatter(S_Co_Open_Time,S_Co_Open)

plt.savefig("OpenSystem_SCo.png")
plt.close()




print("Both at same time")

params_init = (Sr,Sb,Sb,Sg1,np.log10(S_Mono_Open[0]),np.log10(S_Co_Open[0]))
objective = MakeObjective_OffTimes(MonoCulture_Activity_Open,[S_Mono_Open,S_Co_Open],
        [S_Mono_Open[0],S_Co_Open[0]],[S_Mono_Open_Time,S_Co_Open_Time])
results = least_squares(objective,params_init,ftol=1e-10,f_scale=0.05,loss='soft_l1')

print(10**results.x)

Sr,Sb1,Sb2,Sg1,SMono0,SCo0 = results.x
ICs = [10**SMono0]

#Mono
params = (Sr,Sb1,Sg1)
Fitted_Mono = solve_ivp(MonoCulture_Activity_Open,[0,S_Mono_Open_Time[-1]],ICs,args=tuple(params),
                t_eval=S_Mono_Open_Time)

#Co
ICs = [10**SCo0]
params = (Sr,Sb2,Sg1)
Fitted_Co = solve_ivp(MonoCulture_Activity_Open,[0,S_Co_Open_Time[-1]],ICs,args=tuple(params),
                t_eval=S_Co_Open_Time)


plt.figure()

plt.semilogy(Fitted_Mono.t,Fitted_Mono.y[0])
plt.scatter(S_Mono_Open_Time,S_Mono_Open)

plt.semilogy(Fitted_Co.t,Fitted_Co.y[0])
plt.scatter(S_Co_Open_Time,S_Co_Open)


plt.savefig("OpenSystem_BOTH.png")
plt.close()


##############################################################################
#Pseudomonas open
P_Open = [157,161.5,172,169,181,213,292.5,292,235,191.5,168,156,149,138,117,113,115,110.5,111.5,102.5,109,108.5,125,119,117.5,135.5,134.5]

P_Open_Time = [0,0.4,0.7,1,1.3,1.5,2,2.3,2.8,3.1,3.4,3.5,3.8,4.1,4.5,4.9,5.2,5.6,5.9,6.3,6.6,7,7.4,7.7,8.1,8.8,9.4]


Pg1 = 3.8
Pb = 150
params_init = (np.log10(Pr),np.log10(Pb),np.log10(Pg1),-2,0,np.log10(P_Open[0]))


lower = [params_init[0] -1,0,-2,-3,-1,params_init[-1]-1]
upper = [params_init[0] + 1,np.log10(300),np.log10(5),3,3,params_init[-1]+1]

objective = MakeObjective_POffTime(MonoCulture_Activity_POpen,[P_Open],
        [P_Open[0]],P_Open_Time)
results = least_squares(objective,params_init,ftol=1e-10,f_scale=1,loss='soft_l1',
        bounds = [lower,upper])

print(10**results.x)


Pr,Pb,Pg1,Ex,Ec,P0 = results.x
ICs = [10**P0,0]
params = (Pr,Pb,Pg1,Ex,Ec)
"""

ICs = [10**2.221,1]
params = [0.055,5.4,1.03,-2.165,-0.108]
"""
Fitted_P = solve_ivp(MonoCulture_Activity_POpen,[0,P_Open_Time[-1]],ICs,args=tuple(params),
                t_eval=P_Open_Time)

plt.figure()
plt.semilogy(Fitted_P.t,Fitted_P.y[0])
plt.scatter(P_Open_Time,P_Open)




plt.savefig("OpenSystem_P.png")
plt.close()



plt.figure()
plt.semilogy(Fitted_P.t,Fitted_P.y[0])
plt.scatter(P_Open_Time,P_Open)

plt.semilogy(Fitted_Mono.t,Fitted_Mono.y[0])
plt.scatter(S_Mono_Open_Time,S_Mono_Open)

plt.semilogy(Fitted_Co.t,Fitted_Co.y[0])
plt.scatter(S_Co_Open_Time,S_Co_Open)


plt.savefig("OpenSystem_ALL.png")
plt.close()







#Co-fitting P.aeruginosa in monoculture in open and closed
print("Fitting Pmono open and closed at same time")
def MakeObjective_POffTime_ClosedOpen(Model_Closed,Model_Open,datas,time):

    def objective(params):
        alpha,b_closed,b_open,g1,g2,T,Ex,Ec,X0_Closed,X0_Open = params


        #ICs are Clsed X(0), Open X(0), and Open Q(0)
        ICs = [10**X0_Closed]
        params= (alpha,b_closed,g1,g2,T)

        result_Closed = solve_ivp(Model_Closed,[0,time[0][-1]],ICs,args=tuple(params),
                t_eval=time[0])

        params = (alpha,b_open,g1,Ex,Ec)
        ICs = [10**X0_Open,0]

        result_Open = solve_ivp(Model_Open,[0,time[1][-1]],ICs,args=tuple(params),
                t_eval=time[1])


        output = np.concatenate((result_Closed.y[0],result_Open.y[0]))
        output[output<0] = 1e-10

        return np.log10(output) - np.log10(np.concatenate((datas[0],datas[1])))

    return objective



params_init = (Pr,Pb,Pb,Pg1,np.log10(Pg2),np.log10(PT),Ex,Ec,np.log10(P_Mono[0]),np.log10(P_Open[0]))

objective = MakeObjective_POffTime_ClosedOpen(MonoCulture_Activity,MonoCulture_Activity_POpen,[P_Mono,P_Open],
        [np.arange(len(P_Mono)),P_Open_Time])
results = least_squares(objective,params_init,ftol=1e-10,f_scale=1,loss='soft_l1')#,
#        bounds = [lower,upper])

print(10**results.x)



Pr,PbClosed,PbOpen,Pg1,Pg2,PT,Ex,Ec,X0_Closed,X0_Open = results.x

Params_Closed = (Pr,PbClosed,Pg1,Pg2,PT)
ICs = [10**X0_Closed]

result_Closed = solve_ivp(MonoCulture_Activity,[0,len(P_Mono)],ICs,args=tuple(Params_Closed),
                t_eval=np.linspace(0,len(P_Mono),100))#np.arange(len(P_Mono)))


plt.figure()
plt.semilogy(result_Closed.t,result_Closed.y[0],label='With Open')
plt.scatter(np.arange(len(P_Mono)),P_Mono)

plt.semilogy(Fitted_P_Mono.t,Fitted_P_Mono.y[0],label='By Itself')

plt.legend()

plt.savefig("OpenandClosed_Closed.png")
plt.close()




Params_Open = (Pr,PbOpen,Pg1)
ICs = [10**X0_Open,0]

result_Open = solve_ivp(MonoCulture_Activity_POpen,[0,P_Open_Time[-1]],ICs,args=tuple(params),
                t_eval=np.linspace(0,P_Open_Time[-1],100))#P_Open_Time)

plt.figure()
plt.semilogy(result_Open.t,result_Open.y[0],label='With Closed')
plt.scatter(P_Open_Time,P_Open)

plt.semilogy(Fitted_P.t,Fitted_P.y[0],label='By Itself')

plt.legend()

plt.savefig("OpenandClosed_Open.png")
plt.close()











##############################################################################
endtime = time.time()


timetaken = endtime-starttime

print("Time Taken:",timetaken)
