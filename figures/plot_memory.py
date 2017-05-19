import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages


"""
Memory is queried from the App usage info activity.
Only the maximam memory usage is reported.
miniBatch size changes from 1 - 8
"""

# DATA

# free_memory_baseline = (255608 + 1124000) / 1000.0

free_memory_baseline = (441524 + 1071696) / 1000.0

# conv1x1_b1 = free_memory_baseline - (236524 + 1136376) / 1000.0
conv1x1_b1 = free_memory_baseline - (249220 + 1136376) / 1000.0
conv1x1_b2 = free_memory_baseline - (192868 + 1132812) / 1000.0
conv1x1_b4 = free_memory_baseline - (54212 + 1134880) / 1000.0
conv1x1_b8 = free_memory_baseline - (54832 + 1011424) / 1000.0
conv1x2_b1 = free_memory_baseline - (57812 + 1013640) / 1000.0

print conv1x1_b1, conv1x1_b2, conv1x1_b4, conv1x1_b8, conv1x2_b1

memory_JVM = [32.778, 44.508, 47.848, 57.156, 32.778]
memory_nativeAllocate = [conv1x1_b1, conv1x1_b2,
                         conv1x1_b4, conv1x1_b8, conv1x2_b1]

# PLOT
N = 5
width = 0.35
ind = np.arange(N)

p1 = plt.bar(ind, memory_nativeAllocate, width, color='r', hatch='\\')
p2 = plt.bar(ind, memory_JVM, width, color='g',
             hatch='/', bottom=memory_nativeAllocate)


plt.ylabel('Mobile memory usage (MB)')
plt.xlabel('Configurations in layers and batch sizes')
plt.yticks(np.arange(0, 500, 100), rotation='vertical')
plt.xticks(ind, ('Conv1_1\nbatch=1', 'Conv1_1\nbatch=2',
                 'Conv1_1\nbatch=4', 'Conv1_1\nbatch=8', 'Conv1_2\nbatch=1'))
plt.xlim(-width, ind[-1] + 2 * width)

plt.legend((p1[0], p2[0]), ('Native allocated', 'JVM heap size'), loc='best')

matplotlib.rcParams.update({'font.size': 20})
plt.tight_layout()

savefile = PdfPages('./fig_mobile_memory.pdf')
plt.savefig(savefile, format='pdf')
savefile.close()
