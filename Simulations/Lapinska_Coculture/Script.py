#Generic Imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import time
starttime = time.time()


#Import my functions and Data
import sys
sys.path.insert(0,'../Core')

import Models
import Objectives
from Data import *


"""
#Initial Params, from 1 closed monoculture

Sr = np.log10(1.85)
Sb = np.log10(28.1)
Sg1 = np.log10(0.48)
Sg2 = np.log10(1.64 * 10**-2)
ST = np.log10(3.39)
SI0 = np.log10(3.57 * 10**-2)

Pr = np.log10(1.18)
Pb = np.log10(47.9)
Pg1 = np.log10(3.65)
Pg2 = np.log10(2.94 * 10**-2)
PT = np.log10(6.39)
PI0 = np.log10(0.93)
"""

#############################################################################
#Obtain starting parameters from Fig1                                       #
output_params = np.load("../Fig1/output_params.npz")                        #
                                                                            #
Sr = output_params["Sr"]                                                    #
Sb = output_params["Sb"]                                                    #
Sg1 = output_params["Sg1_mono"]                                             #
Sg2 = output_params["Sg2"]                                                  #
ST = output_params["ST"]                                                    #
SI0 = output_params["SI0_closed"]                                           #
                                                                            #
Pr = output_params["Pr"]                                                    #
Pb = output_params["Pb"]                                                    #
Pg1 = output_params["Pg1"]                                                  #
Pg2 = output_params["Pg2"]                                                  #
PT = output_params["PT"]                                                    #
PI0 = output_params["PIClosed"]                                             #
#############################################################################




kl = -10

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
    params_init = (np.asarray([Sr,Sb,Sg1,Sg2,ST,SI0,Pr,Pb,Pg1,Pg2,PT,PI0,kl,np.log10(S_Co[0]),np.log10(P_Co[0])]))
    #print(params_init)
    lowerbounds = [0,0,-3,-3,np.log10(3),-3,    0,0,-2,-2,np.log10(3),-3,   -13,    np.log10(S_Co[0])-1,np.log10(P_Co[0])-1]
    upperbounds = [np.log10(2),3,1,0,1.1,0,       np.log10(2),3,1,0,1.1,0,      -8,      np.log10(S_Co[0])+1,np.log10(P_Co[0])+1]

    """
    for j in range(len(params_init)):
        print(lowerbounds[j],params_init[j],upperbounds[j])
    """
    if DATA == "S_RYC157_P_RYC157":
        #Change in this case as the data is sso evidently different
        params_init[6] = -1
        lowerbounds[6] = -2
        params_init[10] = np.log10(30)
        upperbounds[10] = 2


    objective = Objectives.MakeObjective_Coculture(Models.CoCulture_Independently,[S_Co,P_Co],[S_Co[0],P_Co[0],0])
    results = least_squares(objective,params_init,ftol=1e-10,f_scale=1,loss='soft_l1',bounds=(lowerbounds,upperbounds))


    
    #print(10**results.x)
    print("r",10**results.x[0],10**results.x[6])
    print("b",10**results.x[1],10**results.x[7])
    print("g1",10**results.x[2],10**results.x[8])
    print("g2",10**results.x[3],10**results.x[9])
    print("T",10**results.x[4],10**results.x[10])
    print("I0",10**results.x[5],10**results.x[11])
    print("X0",10**results.x[13],10**results.x[14])
    print("kl",10**results.x[12])

    ICs = [10**results.x[-2],10**results.x[-1],0]
    Fitted = solve_ivp(Models.CoCulture_Independently,[0,len(S_Co)],ICs,args=tuple(results.x[:-2]),
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
    plt.xlabel("time")
    plt.ylabel("bacteria count")

    plt.savefig("Coculture_Independently_" +str(DATA) + ".png")
    plt.close()

endtime=time.time()

print("Time taken:",endtime-starttime)
