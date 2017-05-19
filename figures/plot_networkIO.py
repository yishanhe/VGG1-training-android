
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages


conv1x1_b1 = 12.845

conv1x1_b2 = 25.69

conv1x1_b4 = 51.38

conv1x1_b8 = 102.76

conv1x2_b1 = 12.845

network_io = [conv1x1_b1, conv1x1_b2, conv1x1_b4, conv1x1_b8, conv1x2_b1]

N = 5
width = 0.35
ind = np.arange(N)

p1 = plt.bar(ind, network_io, width, color='b', hatch='\\')

plt.ylabel('Communication cost per iteration (MB)')
plt.xlabel('Configurations in layers and batch sizes')
plt.yticks(np.arange(0, 120, 20), rotation='vertical')
plt.xticks(ind, ('Conv1_1\nbatch=1', 'Conv1_1\nbatch=2',
                 'Conv1_1\nbatch=4', 'Conv1_1\nbatch=8', 'Conv1_2\nbatch=1'))
plt.xlim(-width, ind[-1] + 2 * width)

# plt.legend(p1[0], 'Comm. cost', loc='best')

matplotlib.rcParams.update({'font.size': 20})
plt.tight_layout()

savefile = PdfPages('./fig_mobile_comm.pdf')
plt.savefig(savefile, format='pdf')
savefile.close()
