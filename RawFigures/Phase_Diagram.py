import numpy as np
import matplotlib.pyplot as plt

# Domain
alpha = np.linspace(0, 1, 500)
E_Q = 4 * alpha



width = 40/25.4#40 / 25.4
height = 40/25.4#30/25.4
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



plt.xticks(fontsize=7, fontname='Arial')
plt.yticks(fontsize=7, fontname='Arial')


# Optional legend
#ax.legend(frameon=False)

plt.savefig("Phase_Diagram.png",bbox_inches='tight', dpi=300)

