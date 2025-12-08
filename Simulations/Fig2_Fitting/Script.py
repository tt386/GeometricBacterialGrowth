#Generic Imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.optimize import newton

#Import my functions and Data
import sys
sys.path.insert(0,'../Core')

import Models
import Objectives
from Data import *


##############################################################################
##############################################################################
##############################################################################


#Fitting of S.aureus

#Log10 of the Params
Param_Names = ("alpha", 
"B","I0","I1","BI0",
"g1_mono","g1_co","g2","T",
"MonoClosed init", "MonoOpen Init", "CoOpen Init")

#Observed:
ST = 4  #Last point of linearity

#Mono closed
tlag_closed = 1     #Time at which the lag ends
lagwidth_closed = 1#2#1 #Approximate width of the lag

#Mono_open
tlag_monoopen = 0.1
lagwidth_monoopen = 2#1

#Co_open
tlag_open = 6
lagwidth_open = 4

tendlag_closed =  8 - ST        #Time when saturation phase starts, minus T
endlagwidth_closed = 4          #Width of the saturation transition



SMonoClosed0 = S_Mono[0]
SMono0 = S_Mono_Open[0]
SCo0 = S_Co_Open[0]


#Parameter estimation:
Sr = np.max(np.diff(np.log(S_Mono)))
Sg1_mono = 1/lagwidth_closed * np.log((3 + np.sqrt(5))/(3 - np.sqrt(5)))
Sg1_co = 1/lagwidth_open * np.log((3 + np.sqrt(5))/(3 - np.sqrt(5)))
Sg2 = 1/endlagwidth_closed * np.log((3+np.sqrt(5))/ (3 - np.sqrt(5)))

Sb = np.exp(Sg2 * tendlag_closed)
SI0_closed = np.exp(tlag_closed * Sg1_mono) / Sb
SI0_open =  np.exp(tlag_monoopen * Sg1_mono) / Sb
SbI0 = np.exp(tlag_open * Sg1_co)


params_init = np.log10(np.asarray([Sr,
        Sb,SI0_closed,SI0_open,SbI0,
        Sg1_mono,Sg1_co,Sg2,
        ST,
        SMonoClosed0,SMono0,SCo0]))

lower= (0,   
        0,-5,-5,-5,
        -3,-3,-5,0,
        -np.inf,-np.inf,-np.inf)
upper= (0.3,
        4,0,0,5,
        1,1,0,
        1,
        np.inf,np.inf,np.inf)
"""
for i in range(len(params_init)):
    print(Param_Names[i],lower[i],params_init[i],upper[i],sep='\t')
"""
"""
params_init = (np.log10(1.8),   
        2,-2,-1,1,  
        -1,-1,-3,np.log10(4), 
        np.log10(S_Mono[0]),np.log10(S_Mono_Open[0]),np.log10(S_Co_Open[0]))

lower= (0,   
        0,-5,-5,-5,
        -3,-3,-5,0,
        -np.inf,-np.inf,-np.inf)
upper= (0.3,
        4,0,0,5,
        1,1,0,1,
        np.inf,np.inf,np.inf)
"""

objective = Objectives.MakeObjective_ClosedOpen_Saureus(
        Models.MonoCulture_Closed,Models.MonoCulture_Open_Saureus,[S_Mono,S_Mono_Open,S_Co_Open],
        [np.arange(len(S_Mono)),S_Mono_Open_Time,S_Co_Open_Time])
results = least_squares(objective,params_init,ftol=1e-14,f_scale=1,loss='soft_l1',
        bounds=[lower,upper],
        max_nfev=100000)

#print("ALL S MONO")
#print(10**results.x)

Sr,Sb,SI0_closed,SI0_open,SbI0,Sg1_mono,Sg1_co,Sg2,ST,SMonoClosed0,SMono0,SCo0 = results.x

ICs = [10**SMonoClosed0]

print("Compare initial guess to response for Saureus")
for i in range(len(params_init)):
    print(Param_Names[i],10**params_init[i],10**results.x[i],sep='\t')


#MonoClosed
#params = (Sr,Sb0,Sg1,Sg2,ST,SI0)
params = (Sr,Sb,Sg1_mono,Sg2,ST,SI0_closed)
Fitted_Mono_Closed = solve_ivp(Models.MonoCulture_Closed,[0,len(S_Mono)],ICs,args=tuple(params),
        t_eval=np.arange(len(S_Mono)))

ICs = [10**SMono0]
#Mono
#params = (Sr,Sb1,Sg1)
#params = (Sr,Sb+SI0_open,Sg1_mono)
params = (Sr,Sb,Sg1_mono,SI0_open)

Fitted_Mono = solve_ivp(Models.MonoCulture_Open_Saureus,[0,S_Mono_Open_Time[-1]],ICs,args=tuple(params),
                t_eval=S_Mono_Open_Time)

#Co
ICs = [10**SCo0]
#params = (Sr,Sb2,Sg1)
params = (Sr,SbI0,Sg1_co)
Fitted_Co = solve_ivp(Models.MonoCulture_Open_Saureus,[0,S_Co_Open_Time[-1]],ICs,args=tuple(params),
                t_eval=S_Co_Open_Time)





width = 80/25.4#40 / 25.4
height = 40/25.4#30/25.4
fig, ax1 = plt.subplots(figsize=(width, height))

s = 10
# LEFT AXIS: Closed system
ax1.semilogy(Fitted_Mono_Closed.t, Fitted_Mono_Closed.y[0], linewidth=2,color='k')
ax1.scatter(np.arange(len(S_Mono)),S_Mono, marker='s', s=s, color='none',edgecolors='k', zorder=10,alpha = 0.5)

ax1.set_yticks([1e6, 1e7, 1e8, 1e9, 1e10])
ax1.set_yticklabels([r'$10^6$', r'$10^7$', r'$10^8$', r'$10^9$', r'$10^{10}$'])
ax1.set_ylim(1e6, 1e10)

# X axis
ax1.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
ax1.set_xticklabels([r'$0$', r'$2$', r'$4$', r'$6$', r'$8$', r'$10$', r'$12$', r'$14$'])

ax1.tick_params(which='both',direction='in')


plt.xticks(fontsize=7, fontname='Arial')
plt.yticks(fontsize=7, fontname='Arial')


# RIGHT AXIS: Open system
ax2 = ax1.twinx()

ax2.semilogy(Fitted_Mono.t,Fitted_Mono.y[0], linewidth=2,linestyle='dashed',color='k')
ax2.scatter(S_Mono_Open_Time,S_Mono_Open, marker='o', s=s, color='none',edgecolors='k', zorder=10,alpha=0.5)

ax2.semilogy(Fitted_Co.t,Fitted_Co.y[0], linewidth=2,linestyle='dashed',color='gray')
ax2.scatter(S_Co_Open_Time,S_Co_Open, marker='o', s=s, color='none',edgecolors='gray', zorder=10,alpha=0.5)


ax2.set_yticks([1e0, 1e1, 1e2, 1e3])
ax2.set_yticklabels([r'$10^0$',r'$10^1$', r'$10^2$', r'$10^3$'])
ax2.set_ylim(1, 1e3)

ax2.tick_params(which= 'both',direction='in')

plt.xticks(fontsize=7, fontname='Arial')
plt.yticks(fontsize=7, fontname='Arial')

plt.savefig("All_S_Mono.png", bbox_inches='tight', dpi=300)
plt.close()



print("S Open Mono:",*Fitted_Mono.y[0],sep=',')

print("S Open Co:",*Fitted_Co.y[0],sep=',')


##############################################################################
##############################################################################
##############################################################################

print("P. aeruginosa fitting")

#Params:
Param_Names = ("alpha",
"b","IOpen","IClosed",
"g1","g2","T",
"Ex","Ec",
"X0_Closed","X0_Open")


#Observed:
PT = 6.5#7

#Mono closed
tlag_closed = 1.5
lagwidth_closed = 2#1


#Co_open
tlag_open = 1
lagwidth_open = 1


tendlag_closed =  8 - PT
endlagwidth_closed = 0.5



PX0_Closed = P_Mono[0]
PX0_Open = P_Open[0]

Period = 8
Stable  = 100


#Parameter estimation:
Pr = np.max(np.diff(np.log(P_Mono)))
Pg1 = 1/lagwidth_closed * np.log((3 + np.sqrt(5))/(3 - np.sqrt(5)))
Pg2 = 1/endlagwidth_closed * np.log((3+np.sqrt(5))/ (3 - np.sqrt(5)))

Pb = np.exp(Pg2 * tendlag_closed)
PIClosed = np.exp(tlag_closed * Pg1) / Pb
PIOpen =  np.exp(tlag_open * Pg1) / Pb

Ec = (4*Pr - np.sqrt(16*Pr**2 - 64*np.pi**2/Period**2))/2
Ex = Ec * Pr/Stable


params_init = np.log10(np.asarray([Pr,
        Pb,PIOpen,PIClosed,
        Pg1,Pg2,PT,
        Ex,Ec,
        PX0_Closed,PX0_Open]))


"""
params_init = (np.log10(1.1),
        1.5,0,-1,
        0,-2,np.log10(6),
        np.log10(7.2e-3),np.log10(8e-1),
        np.log10(P_Mono[0]),np.log10(P_Open[0]))
"""
lower = (-1,
        0,-5,-5,
        -4,-4,0,
        -5,-5,
        0,params_init[-1]-0.1)
upper = (0.5,
        3,0,0,
        1,1,1,
        1,1,
        np.inf,np.inf)

"""
for i in range(len(params_init)):
    print(lower[i],params_init[i],upper[i])
"""


objective = Objectives.MakeObjective_ClosedOpen_Paeruginosa(
        Models.MonoCulture_Closed,Models.MonoCulture_Open_Paeruginosa,
        [P_Mono,P_Open],
        [np.arange(len(P_Mono)),P_Open_Time])
results = least_squares(objective,params_init,ftol=1e-14,f_scale=1,
        loss='linear',
        bounds = [lower,upper])

#print(10**results.x)

print("Compare initial guess to response for Saureus")
for i in range(len(params_init)):
    print(Param_Names[i],10**params_init[i],10**results.x[i],sep='\t')

Pr,Pb,PIOpen,PIClosed,Pg1,Pg2,PT,Ex,Ec,PX0_Closed,PX0_Open = results.x


Params_Closed = (Pr,Pb,Pg1,Pg2,PT,PIClosed)
ICs = [10**PX0_Closed]

result_Closed = solve_ivp(Models.MonoCulture_Closed,[0,len(P_Mono)],ICs,args=tuple(Params_Closed),
                t_eval=np.arange(len(P_Mono)))



#Params_Open = (Pr,Pb+PIOpen,Pg1,Ex,Ec)
Params_Open = (Pr,Pb,Pg1,Ex,Ec,PIOpen)

ICs = [10**PX0_Open,0]

result_Open = solve_ivp(Models.MonoCulture_Open_Paeruginosa,[0,P_Open_Time[-1]],ICs,args=tuple(Params_Open),
                t_eval=P_Open_Time)






width = 80/25.4#40 / 25.4
height  = 40/25.4#30/25.4
fig, ax1 = plt.subplots(figsize=(width, height))

s = 10
# LEFT AXIS: Closed system
ax1.semilogy(result_Closed.t, result_Closed.y[0], linewidth=2,color='k')
ax1.scatter(result_Closed.t[:len(P_Mono)], P_Mono, marker='s', s=s, color='none',edgecolors='k', zorder=10,alpha=0.5)

ax1.set_yticks([1e6, 1e7, 1e8, 1e9, 1e10])
ax1.set_yticklabels([r'$10^6$', r'$10^7$', r'$10^8$', r'$10^9$', r'$10^{10}$'])
ax1.set_ylim(1e6, 1e10)

# X axis
ax1.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
ax1.set_xticklabels([r'$0$', r'$2$', r'$4$', r'$6$', r'$8$', r'$10$', r'$12$', r'$14$'])

ax1.tick_params(which='both',direction='in')

plt.xticks(fontsize=7, fontname='Arial')
plt.yticks(fontsize=7, fontname='Arial')


# RIGHT AXIS: Open system
ax2 = ax1.twinx()
ax2.semilogy(result_Open.t, result_Open.y[0], linewidth=2,linestyle='dashed',color='k')
ax2.scatter(P_Open_Time, P_Open, marker='o', s=s, color='none',edgecolors='k', zorder=10,alpha=0.5)

ax2.set_yticks([1e1, 1e2, 1e3])
ax2.set_yticklabels([r'$10^1$', r'$10^2$', r'$10^3$'])
ax2.set_ylim(10, 1e3)

ax2.tick_params(which='both',direction='in')


plt.xticks(fontsize=7, fontname='Arial')
plt.yticks(fontsize=7, fontname='Arial')

plt.savefig("All_P_Mono_Fitting.png", bbox_inches='tight', dpi=300)
plt.close()


print("P Open:",*P_Open,sep=',')

##############################################################################
##############################################################################
##############################################################################



#Save Data
outputfilename = "output_params.npz"

np.savez(outputfilename,
        Sr=Sr,
        Sb=Sb,
        SI0_closed=SI0_closed,
        SI0_open=SI0_open,
        SbI0=SbI0,
        Sg1_mono=Sg1_mono,
        Sg1_co=Sg1_co,
        Sg2=Sg2,
        ST=ST,
        SMonoClosed0=SMonoClosed0,
        SMono0=SMono0,
        SCo0=SCo0,

        Pr=Pr,
        Pb=Pb,
        PIOpen=PIOpen,
        PIClosed=PIClosed,
        Pg1=Pg1,
        Pg2=Pg2,
        PT=PT,
        Ex=Ex,
        Ec=Ec,
        PX0_Closed=PX0_Closed,
        PX0_Open=PX0_Open)
