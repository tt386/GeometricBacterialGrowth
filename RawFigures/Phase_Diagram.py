import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp










##############################################################################
# Different solutions
##############################################################################

# ODE system
def system(t, y, a):
    X, Q = y
    dXdt = a*X - X*Q
    dQdt = X - Q
    return [dXdt, dQdt]

# Parameters
alphas = [0.1, 0.25, 0.7]
t_span = (0, 20)
t_eval = np.linspace(*t_span, 2000)
y0 = [1,0]

# Plot
width = 80/25.4#40 / 25.4
height = 80/25.4#30/25.4
fig, ax = plt.subplots(figsize=(width,height))

for a in alphas:
    sol = solve_ivp(system, t_span, y0, t_eval=t_eval, args=(a,))
    ax.plot(sol.t, sol.y[0], label=rf'$X(t),\ \alpha={a}$',linewidth=2,color='k',alpha=min(a*2,1))

    ax.plot(sol.t, np.ones(len(sol.y[0])) * a,linewidth=1,color='k',linestyle='dashed')


# Labels
#ax.set_xlabel(r'$t$')
#ax.set_ylabel(r'$X(t)$')
#ax.legend(frameon=False)


ax.set_xticks([0,10,20])
ax.set_xticklabels([r'$0$',r'$10$' ,r'$20$'])


# Existing ticks
yticks = [0, 1]

# Add alpha ticks
yticks_extended = yticks + alphas

# Corresponding labels
yticklabels = (
    [r'$0$', r'$1$'] +
    [rf'${a}$' for a in alphas]
)

ax.set_yticks(yticks_extended)
ax.set_yticklabels(yticklabels)
from matplotlib.ticker import FormatStrFormatter

ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


plt.xticks(fontsize=7, fontname='Arial')
plt.yticks(fontsize=7, fontname='Arial')



plt.savefig("Example_Solutions.png",bbox_inches='tight', dpi=300)



##############################################################################
# Phase Diagram
##############################################################################

# Domain
alpha = np.linspace(0, 1, 500)
E_Q = 4 * alpha



width = 80/25.4#40 / 25.4
height = 80/25.4#30/25.4
fig, ax = plt.subplots(figsize=(width,height))


# Fill regions
ax.fill_between(alpha, E_Q, y2=E_Q.max(),
                color='red', alpha=0.4, label=r'$E_Q > 4\alpha$')

ax.fill_between(alpha, 0, E_Q,
                color='blue', alpha=0.4, label=r'$E_Q < 4\alpha$')

# Phase boundary
ax.plot(alpha, E_Q, color='black', linewidth=3)

"""
# Labels
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$E_Q$')
"""

# Limits (adjust as needed)
ax.set_xlim(alpha.min(), alpha.max())
ax.set_ylim(0, E_Q.max())


ax.set_xticks([0,0.5,1])
ax.set_xticklabels([r'$0$',r'$0.5$' ,r'$1$'])

ax.set_yticks([0,2,4])
ax.set_yticklabels([r'$0$',r'$2$', r'$4$'])



plt.xticks(fontsize=7, fontname='Arial')
plt.yticks(fontsize=7, fontname='Arial')


# Optional legend
#ax.legend(frameon=False)

plt.savefig("Phase_Diagram.png",bbox_inches='tight', dpi=300)

