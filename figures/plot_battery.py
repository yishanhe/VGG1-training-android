#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

conv1x1_b1 = [0, 3, 8, 13, 16]
conv1x1_b2 = [0, 10, 22, 34, 45]
conv1x1_b4 = [0, 13, 25, 38, 50]
conv1x1_b8 = [0, 20, 37, 52, 68]
conv1x2_b1 = [0, 25, 43, 65, 81]

fig = plt.figure()
x = [0, 5, 10, 15, 20]
plt.plot(x, conv1x1_b1, 'b^-',
         label='Conv1_1, batch=1', linewidth=2, ms=10)
plt.plot(x, conv1x1_b2, 'ro-',
         label='Conv1_1, batch=2', linewidth=2, ms=10)
plt.plot(x, conv1x1_b4, 'bs-',
         label='Conv1_1, batch=4', linewidth=2, ms=10)
plt.plot(x, conv1x1_b8, 'yv-',
         label='Conv1_1, batch=8', linewidth=2, ms=10)
plt.plot(x, conv1x2_b1, 'kd-',
         label='Conv1_2, batch=1', linewidth=2, ms=10)


plt.legend(loc='best', labelspacing=0.05)
plt.ylabel('Battery consumption (mAh)')
plt.xlabel('Time (min)')
plt.xlim([0, 23])
plt.ylim([0, 85])


matplotlib.rcParams.update({'font.size': 20})
plt.tight_layout()

savefile = PdfPages('./fig_mobile_energy.pdf')
plt.savefig(savefile, format='pdf')
savefile.close()
