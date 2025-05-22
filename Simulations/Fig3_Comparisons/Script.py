#Generic Imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


#Import my functions and Data
import sys
sys.path.insert(0,'../Core')

import Models
import Objectives
from Data import *

#############################################################################
#Obtain starting parameters from Fig1                                       #
output_params = np.load("../Fig2_Fitting/output_params.npz")                #
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

#Loss function
lossfun = 'linear'#'soft_l1'
f_scale = 1#0.05

DATA = ['S_Mono','P_Mono']

for i in DATA:
    if i == 'S_Mono':
        data = S_Mono
    elif i =='P_Mono':
        data = P_Mono

    print("Parameter fitting for ",i)

    ##############################################################################
    ##############################################################################
    ##############################################################################

    #Naive Isolated
    #Fitting of S.aureus
    if i == 'S_Mono':
        #params_init = (np.log10(1.85),np.log10(28),np.log10(0.48),np.log10(1.64*10**-2),np.log10(3.39),3.57*10**-2,np.log10(data[0]))
        params_init = (Sr,Sb,Sg1,Sg2,ST,SI0,np.log10(data[0]))
    elif i == 'P_Mono':
        params_init = (Pr,Pb,Pg1,Pg2,PT,PI0,np.log10(data[0]))
        #params_init = (np.log10(1.18),np.log10(48),np.log10(3.65),np.log10(2.94*10**-2),np.log10(6.39),np.log10(0.93),np.log10(data[0]))


    objective = Objectives.MakeObjective(
            Models.MonoCulture_Closed,[data],[0])

    results = least_squares(objective,params_init,ftol=1e-14,f_scale=f_scale,loss=lossfun)#,
    #        bounds=[lower,upper])

    #Sr,Sb,I0_closed,I0_open,bI0,Sg1_mono,Sg1_co,Sg2,ST,SMonoClosed0,SMono0,SCo0 = results.x

    Sr,Sb,Sg1_mono,Sg2,ST,I0_closed,SMonoClosed0 = results.x

    print("Geometric Fitting",10**results.x)
    print("Geometric Cost",results.cost)
    ICs = [10**SMonoClosed0]

    #MonoClosed
    #params = (Sr,Sb0,Sg1,Sg2,ST,SI0)
    params = (Sr,Sb,Sg1_mono,Sg2,ST,I0_closed)
    Fitted_Mono_Closed = solve_ivp(Models.MonoCulture_Closed,[0,len(data)],ICs,args=tuple(params),
            t_eval=np.arange(len(data)))


    
    #Param Estimations
    X0 = data[0]
    XF= data[-1]
    if i == "S_Mono":
        t0 = 8
    elif i == "P_Mono":
        t0 = 6
    params_init = [np.log10(1.5),np.log10(XF),np.log10(t0),np.log10(X0)]

    #Logistic
    print("Logistic")

    #Estimate k:
    #k = -np.gradient((XF-X0)/(data-X0))[t0]
    k = 4/(XF-X0) * np.gradient(data)[t0]
    print("Logistic Estimated k is",k)
    params_init[0] = np.log10(k)
    print("Logistic Initial Guess",10**np.asarray(params_init))
    objective = Objectives.MakeObjective_Fits(Models.Logistic,data)
    S_Mono_Logistic = least_squares(objective,params_init,
            ftol=1e-12,f_scale=f_scale,loss=lossfun)

    print("Logistic Fitting",10**S_Mono_Logistic.x)
    print("Logistic Cost",S_Mono_Logistic.cost)
    Sr_L,SK_L,SIC_L,t0_L = S_Mono_Logistic.x


    #Gompertz
    print("Gompertz")
    k = np.gradient(np.log((data-X0)/(XF-X0)))[t0]
    print("Gompertz estimated k is ",k)
    params_init[0] = np.log10(k)
    print("Gompertz Initial Guess",10**np.asarray(params_init))
    objective = Objectives.MakeObjective_Fits(Models.Gompertz,data)
    S_Mono_Gompertz = least_squares(objective,params_init,
            ftol=1e-12,f_scale=f_scale,loss=lossfun)

    print("Gompertz Fitting",10**S_Mono_Gompertz.x)
    print("Gompertz cost", S_Mono_Gompertz.cost)
    Sr_G,SK_G,SIC_G,t0_G = S_Mono_Gompertz.x


    print("Kinetic")
    params_kinetic = (-9,10,0,np.log10(data[0]))
    print("Kinetic Initial guess:",10**np.asarray(params_kinetic))
    lower = [-10,8,-1,params_kinetic[-1]-1]
    upper = [-8,12,1,params_kinetic[-1]+1]

    objective = Objectives.MakeObjective_Kinetic(Models.Kinetic_Monoculture,[data])
    KineticModel = least_squares(objective,params_kinetic,bounds=(lower,upper),
            ftol=1e-12,f_scale=f_scale,loss=lossfun)

    kc,Y,kx,X0 = KineticModel.x

    print("Kinetic fitting",10**KineticModel.x)
    print("Kinetic Cost",KineticModel.cost)
    params = (kc,Y,kx)

    Fitted_Kinetic = solve_ivp(Models.Kinetic_Monoculture,[0,len(data)],[10**X0,0,0.1],args=tuple(params),
            t_eval=np.arange(len(data)))



    t = np.linspace(0,len(data)-1,100)


    width = 80/25.4#40 / 25.4
    height = 40/25.4#30/25.4
    fig, ax1 = plt.subplots(figsize=(width,height))

    s = 10
    ax1.semilogy(Fitted_Mono_Closed.t,Fitted_Mono_Closed.y[0],
            linewidth = 2,label='Activity',color='k',alpha=1)
    ax1.semilogy(t,Models.Logistic(t,Sr_L,SK_L,SIC_L,t0_L),
            linewidth=2,linestyle='dashed',color='#D81B60',alpha=0.8)
    ax1.semilogy(t,Models.Gompertz(t,Sr_G,SK_G,SIC_G,t0_G),
            linewidth=2,linestyle='dotted',color='#1E88E5',alpha=0.8)
    ax1.semilogy(Fitted_Kinetic.t,Fitted_Kinetic.y[0],
            linewidth=2,linestyle='-.',color='#FFC107',alpha=0.8)

    ax1.scatter(np.arange(len(data)),data,
            s=s,marker='s',color='none',edgecolors='k',zorder=10)

    ax1.set_yticks([1e6, 1e7, 1e8, 1e9, 1e10])
    ax1.set_yticklabels([r'$10^6$', r'$10^7$', r'$10^8$', r'$10^9$', r'$10^{10}$'])
    ax1.set_ylim(1e6, 1e10)

    # X axis
    ax1.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
    ax1.set_xticklabels([r'$0$', r'$2$', r'$4$', r'$6$', r'$8$', r'$10$', r'$12$', r'$14$'])

    ax1.tick_params(which='both',direction='in')


    plt.xticks(fontsize=7, fontname='Arial')
    plt.yticks(fontsize=7, fontname='Arial')

    plt.xticks(fontsize=7, fontname='Arial')
    plt.yticks(fontsize=7, fontname='Arial')

    plt.ylim(1e6,10**(10.5))

    plt.savefig(str(i) + "_Comparisons.png",bbox_inches='tight', dpi=300)
    plt.close()



    """
    #######################################################
    #Comapring fits

    t = np.linspace(0,len(data)-1,100)


    width = 40 / 25.4
    fig, ax1 = plt.subplots(figsize=(width, 30 / 25.4))

    s = 5
    # LEFT AXIS: LOG
    ax1.semilogy(Fitted_Mono_Closed.t,Fitted_Mono_Closed.y[0],
            linewidth = 1,label='Activity',color='k',alpha=0.5)
    ax1.semilogy(t,Models.Logistic(t,Sr_L,SK_L,SIC_L,t0_L),
            linewidth=1,linestyle='dashed',color='k',alpha=0.5)
    ax1.semilogy(t,Models.Gompertz(t,Sr_G,SK_G,SIC_G,t0_G),
            linewidth=1,linestyle='dotted',color='k',alpha=0.5)
    ax1.scatter(np.arange(len(data)),data,
            s=s,color='k',zorder=10)

    ax1.set_yticks([1e6, 1e7, 1e8, 1e9, 1e10])
    ax1.set_yticklabels([r'$10^6$', r'$10^7$', r'$10^8$', r'$10^9$', r'$10^{10}$'])
    ax1.set_ylim(1e6, 1e10)

    # X axis
    ax1.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
    ax1.set_xticklabels([r'$0$', r'$2$', r'$4$', r'$6$', r'$8$', r'$10$', r'$12$', r'$14$'])


    plt.xticks(fontsize=7, fontname='Arial')
    plt.yticks(fontsize=7, fontname='Arial')

    # RIGHT AXIS: Linear
    ax2 = ax1.twinx()

    ax2.plot(Fitted_Mono_Closed.t,Fitted_Mono_Closed.y[0],
            linewidth = 1,label='Activity',color='k',alpha=0.2)
    ax2.plot(t,Models.Logistic(t,Sr_L,SK_L,SIC_L,t0_L),
            linewidth=1,linestyle='dashed',color='k',alpha=0.2)
    ax2.plot(t,Models.Gompertz(t,Sr_G,SK_G,SIC_G,t0_G),
            linewidth=1,linestyle='dotted',color='k',alpha=0.2)
    ax2.scatter(np.arange(len(data)),data,
            s=s,color='k',marker='x',zorder=10)

    ax2.set_yticks([0, 1e9, 2e9, 3e9])
    ax2.set_yticklabels([r'$0$',r'$1$', r'$2$', r'$3$'])
    ax2.set_ylim(0, 3.5 * 10**9)

    plt.xticks(fontsize=7, fontname='Arial')
    plt.yticks(fontsize=7, fontname='Arial')


    plt.savefig(str(i) + "_Comparisons.png",bbox_inches='tight', dpi=300)
    plt.close()

    """



    """
    #Doing Kinetic fits
    #Params: kc,Y,kx, X0
    print("Kinetic Fit")


    params_kinetic = (-9,10,0,np.log10(data[0]))
    print("Kinetic Initial guess:",10**np.asarray(params_kinetic))
    lower = [-10,8,-1,params_kinetic[-1]-1]
    upper = [-8,12,1,params_kinetic[-1]+1]

    objective = Objectives.MakeObjective_Kinetic(Models.Kinetic_Monoculture,[data])
    KineticModel = least_squares(objective,params_kinetic,bounds=(lower,upper),
            ftol=1e-12,f_scale=f_scale,loss=lossfun)

    kc,Y,kx,X0 = KineticModel.x

    print("Kinetic fitting",10**KineticModel.x)
    print("Kinetic Cost",KineticModel.cost)
    params = (kc,Y,kx)

    Fitted_Kinetic = solve_ivp(Models.Kinetic_Monoculture,[0,len(data)],[10**X0,0,0.1],args=tuple(params),
            t_eval=np.arange(len(data)))





    width = 40 / 25.4
    fig, ax1 = plt.subplots(figsize=(width, 30 / 25.4))

    s = 5
    # LEFT AXIS: LOG
    ax1.semilogy(Fitted_Mono_Closed.t,Fitted_Mono_Closed.y[0],
            linewidth = 1,label='Activity',color='k',alpha=0.5)
    ax1.semilogy(Fitted_Kinetic.t,Fitted_Kinetic.y[0],
            linewidth=1,linestyle='-.',color='k',alpha=0.5)
    ax1.scatter(np.arange(len(data)),data,
            s=s,color='k',zorder=10)

    ax1.set_yticks([1e6, 1e7, 1e8, 1e9, 1e10])
    ax1.set_yticklabels([r'$10^6$', r'$10^7$', r'$10^8$', r'$10^9$', r'$10^{10}$'])
    ax1.set_ylim(1e6, 1e10)

    # X axis
    ax1.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
    ax1.set_xticklabels([r'$0$', r'$2$', r'$4$', r'$6$', r'$8$', r'$10$', r'$12$', r'$14$'])


    plt.xticks(fontsize=7, fontname='Arial')
    plt.yticks(fontsize=7, fontname='Arial')

    # RIGHT AXIS: Linear
    ax2 = ax1.twinx()

    ax2.plot(Fitted_Mono_Closed.t,Fitted_Mono_Closed.y[0],
            linewidth = 1,label='Activity',color='k',alpha=0.2)
    ax2.plot(Fitted_Kinetic.t,Fitted_Kinetic.y[0],
            linewidth=1,linestyle='-.',color='k',alpha=0.2)
    ax2.scatter(np.arange(len(data)),data,
            s=s,color='k',marker='x',zorder=10)

    ax2.set_yticks([0, 1e9, 2e9, 3e9])
    ax2.set_yticklabels([r'$0$',r'$1$', r'$2$', r'$3$'])
    ax2.set_ylim(0, 3.5 * 10**9)

    plt.xticks(fontsize=7, fontname='Arial')
    plt.yticks(fontsize=7, fontname='Arial')


    plt.savefig(str(i) + "_KineticComparisons.png",bbox_inches='tight', dpi=300)
    plt.close()
    """
