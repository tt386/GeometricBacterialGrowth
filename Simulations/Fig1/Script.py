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


##############################################################################
##############################################################################
##############################################################################


#Fitting of S.aureus
"""
Log10 of the Params
#alpha, 
B,I0,I1,BI0,
g1_mono,g1_co,g2,T,
MonoClosed init, MonoOpen Init, CoOpen Init
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


objective = Objectives.MakeObjective_ClosedOpen_Saureus(
        Models.MonoCulture_Closed,Models.MonoCulture_Open_Saureus,[S_Mono,S_Mono_Open,S_Co_Open],
        [np.arange(len(S_Mono)),S_Mono_Open_Time,S_Co_Open_Time])
results = least_squares(objective,params_init,ftol=1e-14,f_scale=0.05,loss='soft_l1',
        bounds=[lower,upper],
        max_nfev=100000)

print("ALL S MONO")
print(10**results.x)

Sr,Sb,I0_closed,I0_open,bI0,Sg1_mono,Sg1_co,Sg2,ST,SMonoClosed0,SMono0,SCo0 = results.x

ICs = [10**SMonoClosed0]

#MonoClosed
#params = (Sr,Sb0,Sg1,Sg2,ST,SI0)
params = (Sr,Sb,Sg1_mono,Sg2,ST,I0_closed)
Fitted_Mono_Closed = solve_ivp(Models.MonoCulture_Closed,[0,len(S_Mono)],ICs,args=tuple(params),
        t_eval=np.arange(len(S_Mono)))

ICs = [10**SMono0]
#Mono
#params = (Sr,Sb1,Sg1)
params = (Sr,Sb+I0_open,Sg1_mono)
Fitted_Mono = solve_ivp(Models.MonoCulture_Open_Saureus,[0,S_Mono_Open_Time[-1]],ICs,args=tuple(params),
                t_eval=S_Mono_Open_Time)

#Co
ICs = [10**SCo0]
#params = (Sr,Sb2,Sg1)
params = (Sr,bI0,Sg1_co)
Fitted_Co = solve_ivp(Models.MonoCulture_Open_Saureus,[0,S_Co_Open_Time[-1]],ICs,args=tuple(params),
                t_eval=S_Co_Open_Time)





width = 40 / 25.4
fig, ax1 = plt.subplots(figsize=(width, 30 / 25.4))

s = 5
# LEFT AXIS: Closed system
ax1.semilogy(Fitted_Mono_Closed.t, Fitted_Mono_Closed.y[0], linewidth=1,color='k',zorder=5)
ax1.scatter(np.arange(len(S_Mono)),S_Mono, marker='s', s=s, color='none',edgecolors='k', zorder=2,alpha = 0.5)

ax1.set_yticks([1e6, 1e7, 1e8, 1e9, 1e10])
ax1.set_yticklabels([r'$10^6$', r'$10^7$', r'$10^8$', r'$10^9$', r'$10^{10}$'])
ax1.set_ylim(1e6, 1e10)

# X axis
ax1.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
ax1.set_xticklabels([r'$0$', r'$2$', r'$4$', r'$6$', r'$8$', r'$10$', r'$12$', r'$14$'])


plt.xticks(fontsize=7, fontname='Arial')
plt.yticks(fontsize=7, fontname='Arial')


# RIGHT AXIS: Open system
ax2 = ax1.twinx()

ax2.semilogy(Fitted_Mono.t,Fitted_Mono.y[0], linewidth=1,linestyle='dashed',color='k',zorder=5)
ax2.scatter(S_Mono_Open_Time,S_Mono_Open, marker='o', s=s, color='none',edgecolors='k', zorder=2,alpha=0.5)

ax2.semilogy(Fitted_Co.t,Fitted_Co.y[0], linewidth=1,linestyle='dashed',color='gray',zorder=5)
ax2.scatter(S_Co_Open_Time,S_Co_Open, marker='o', s=s, color='none',edgecolors='gray', zorder=2,alpha=0.5)


ax2.set_yticks([1e0, 1e1, 1e2, 1e3])
ax2.set_yticklabels([r'$10^0$',r'$10^1$', r'$10^2$', r'$10^3$'])
ax2.set_ylim(1, 1e3)

plt.xticks(fontsize=7, fontname='Arial')
plt.yticks(fontsize=7, fontname='Arial')

plt.savefig("All_S_Mono.png", bbox_inches='tight', dpi=300)
plt.close()






##############################################################################
##############################################################################
##############################################################################

print("P. aeruginosa fitting")
"""
Params:
alpha,
b,IOpen,IClosed,
g1,g2,T,
Ex,Ec,
X0_Closed,X0_Open
"""


params_init = (np.log10(1.1),
        1.5,0,-1,
        0,-2,np.log10(6),
        np.log10(7.2e-3),np.log10(8e-1),
        np.log10(P_Mono[0]),np.log10(P_Open[0]))

lower = (-1,
        0,-5,-5,
        -4,-4,0,
        -5,-5,
        0,0)
upper = (0.5,
        3,0,0,
        1,1,1,
        1,1,
        np.inf,np.inf)

objective = Objectives.MakeObjective_ClosedOpen_Paeruginosa(
        Models.MonoCulture_Closed,Models.MonoCulture_Open_Paeruginosa,
        [P_Mono,P_Open],
        [np.arange(len(P_Mono)),P_Open_Time])
results = least_squares(objective,params_init,ftol=1e-14,f_scale=1,
        loss='linear',
        bounds = [lower,upper])

print(10**results.x)


Pr,Pb,IOpen,IClosed,Pg1,Pg2,PT,Ex,Ec,X0_Closed,X0_Open = results.x


Params_Closed = (Pr,Pb,Pg1,Pg2,PT,IClosed)
ICs = [10**X0_Closed]

result_Closed = solve_ivp(Models.MonoCulture_Closed,[0,len(P_Mono)],ICs,args=tuple(Params_Closed),
                t_eval=np.arange(len(P_Mono)))



Params_Open = (Pr,Pb+IOpen,Pg1,Ex,Ec)
ICs = [10**X0_Open,0]

result_Open = solve_ivp(Models.MonoCulture_Open_Paeruginosa,[0,P_Open_Time[-1]],ICs,args=tuple(Params_Open),
                t_eval=P_Open_Time)






width = 40 / 25.4
fig, ax1 = plt.subplots(figsize=(width, 30 / 25.4))

s = 5
# LEFT AXIS: Closed system
ax1.semilogy(result_Closed.t, result_Closed.y[0], linewidth=1,color='k')
ax1.scatter(result_Closed.t[:len(P_Mono)], P_Mono, marker='s', s=s, color='none',edgecolors='k', zorder=5,alpha=0.5)

ax1.set_yticks([1e6, 1e7, 1e8, 1e9, 1e10])
ax1.set_yticklabels([r'$10^6$', r'$10^7$', r'$10^8$', r'$10^9$', r'$10^{10}$'])
ax1.set_ylim(1e6, 1e10)

# X axis
ax1.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
ax1.set_xticklabels([r'$0$', r'$2$', r'$4$', r'$6$', r'$8$', r'$10$', r'$12$', r'$14$'])


plt.xticks(fontsize=7, fontname='Arial')
plt.yticks(fontsize=7, fontname='Arial')


# RIGHT AXIS: Open system
ax2 = ax1.twinx()
ax2.semilogy(result_Open.t, result_Open.y[0], linewidth=1,linestyle='dashed',color='k')
ax2.scatter(P_Open_Time, P_Open, marker='o', s=s, color='none',edgecolors='k', zorder=5,alpha=0.5)

ax2.set_yticks([1e1, 1e2, 1e3])
ax2.set_yticklabels([r'$10^1$', r'$10^2$', r'$10^3$'])
ax2.set_ylim(10, 1e3)

plt.xticks(fontsize=7, fontname='Arial')
plt.yticks(fontsize=7, fontname='Arial')

plt.savefig("All_P_Mono_Fitting.png", bbox_inches='tight', dpi=300)
plt.close()




