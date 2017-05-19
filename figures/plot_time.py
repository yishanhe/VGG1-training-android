import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages


"""
Layers of Conv
Using batch size 1

"""


conv1x1_b1_io = [0.052, 0.051, 0.072, 0.065, 0.074, 0.053]
conv1x1_b1_fw1 = [0.133, 0.145, 0.153, 0.21, 0.18, 0.139]
conv1x1_b1_bw1 = [0.176, 0.17, 0.144, 0.158, 0.166, 0.176]


conv1x1_b2_io = [0.094, 0.091, 0.096, 0.094, 0.094, 0.096]
conv1x1_b2_fw1 = [0.216, 0.219, 0.227, 0.228, 0.213, 0.203]
conv1x1_b2_bw1 = [3.258, 3.065, 3.051, 3.28, 3.104, 3.096]


conv1x1_b4_io = [0.171, 0.18, 0.187, 0.171, 0.173, 0.173]
conv1x1_b4_fw1 = [0.332, 0.399, 0.3, 0.358, 0.328, 0.368]
conv1x1_b4_bw1 = [6.434, 6.504, 6.716, 6.498, 6.34, 6.492]


conv1x1_b8_io = [0.334, 0.336, 0.331, 0.332, 0.335, 0.338]
conv1x1_b8_fw1 = [0.594, 0.577, 0.606, 0.558, 0.59, 0.647]
conv1x1_b8_bw1 = [13.605, 13.516, 13.506, 13.444, 13.532, 13.383]

conv1x2_b1_io = [0.054, 0.055, 0.054, 0.059,
                 0.061, 0.065, 0.059, 0.055, 0.055, 0.057]
conv1x2_b1_fw1 = [0.146, 0.155, 0.164, 0.152,
                  0.235, 0.165, 0.222, 0.133, 0.22, 0.152]
conv1x2_b1_fw2 = [1.329, 1.34, 1.314, 1.337,
                  1.744, 1.306, 1.655, 0.943, 1.677, 0.983]
conv1x2_b1_bw2 = [2.088, 2.078, 2.088, 2.142,
                  2.051, 2.088, 2.056, 2.09, 2.045, 2.136]
conv1x2_b1_bw1 = [0.152, 0.188, 0.147, 0.172,
                  0.164, 0.16, 0.145, 0.156, 0.143, 0.157]


io_mean = [np.mean(conv1x1_b1_io), np.mean(conv1x1_b2_io), np.mean(
    conv1x1_b4_io), np.mean(conv1x1_b8_io), np.mean(conv1x2_b1_io)]
fw1_mean = [np.mean(conv1x1_b1_fw1), np.mean(conv1x1_b2_fw1), np.mean(
    conv1x1_b4_fw1), np.mean(conv1x1_b8_fw1), np.mean(conv1x2_b1_fw1)]
fw2_mean = [0, 0, 0, 0, np.mean(conv1x2_b1_fw2)]
bw2_mean = [0, 0, 0, 0, np.mean(conv1x2_b1_bw2)]
bw1_mean = [np.mean(conv1x1_b1_bw1), np.mean(conv1x1_b2_bw1), np.mean(
    conv1x1_b4_bw1), np.mean(conv1x1_b8_bw1), np.mean(conv1x2_b1_bw1)]

io_std = [np.std(conv1x1_b1_io), np.std(conv1x1_b2_io), np.std(
    conv1x1_b4_io), np.std(conv1x1_b8_io), np.std(conv1x2_b1_io)]
fw1_std = [np.std(conv1x1_b1_fw1), np.std(conv1x1_b2_fw1), np.std(
    conv1x1_b4_fw1), np.std(conv1x1_b8_fw1), np.std(conv1x2_b1_fw1)]
fw2_std = [0, 0, 0, 0, np.std(conv1x2_b1_fw2)]
bw2_std = [0, 0, 0, 0, np.std(conv1x2_b1_bw2)]
bw1_std = [np.std(conv1x1_b1_bw1), np.std(conv1x1_b2_bw1), np.std(
    conv1x1_b4_bw1), np.std(conv1x1_b8_bw1), np.std(conv1x2_b1_bw1)]


N = 5
ind = np.arange(N)
width = 0.5

p1 = plt.bar(ind * 3 - 2 * width, io_mean, width, color='r', yerr=io_std)
p2 = plt.bar(ind * 3 - width, fw1_mean, width, color='g', yerr=fw1_std)
p3 = plt.bar(ind * 3, bw1_mean, width, color='b', yerr=bw1_std)
p4 = plt.bar(ind * 3 + width, fw2_mean, width, color='y', yerr=fw2_std)
p5 = plt.bar(ind * 3 + 2 * width, bw2_mean, width, color='k', yerr=bw2_std)

# p1 = plt.bar(ind, bw1_mean, width, color='r',
#              hatch='\\', yerr=bw1_std, bottom=bw2_mean)
# p2 = plt.bar(ind, bw2_mean, width, color='g',
#              hatch='//', bottom=fw2_mean, yerr=bw2_std)
# p3 = plt.bar(ind, fw2_mean, width, color='b',
#              hatch='/', bottom=fw1_mean, yerr=fw2_std)
# p4 = plt.bar(ind, fw1_mean, width, color='y',
#              hatch='.', bottom=io_mean, yerr=fw1_std)
# p5 = plt.bar(ind, io_mean, width, color='k',
#              hatch='+', bottom=fw1_mean, yerr=io_std)


plt.ylabel('Time cost (s)')
plt.xlabel('Configurations in layers and batch sizes')
plt.yticks(np.arange(0, 15, 5), rotation='vertical')
plt.yscale('log')
plt.xticks(ind * 3, ('Conv1_1\nbatch=1', 'Conv1_1\nbatch=2',
                     'Conv1_1\nbatch=4', 'Conv1_1\nbatch=8', 'Conv1_2\nbatch=1'))
plt.xlim(-3 * width, 3 * ind[-1] + 5 * width)
plt.ylim(0, 200)
plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0]), ('Load batch', 'Conv1_1 forward',
                                                 'Conv1_1 backward', 'Conv1_2 forward', 'Conv1_2 backward'), loc='upper left', markerscale=0.7, labelspacing=0.05, ncol=2, fontsize=18, columnspacing=0.1)

matplotlib.rcParams.update({'font.size': 20})
plt.tight_layout()


savefile = PdfPages('./fig_mobile_time.pdf')
# fig = matplotlib.pyplot.gcf()
# fig.set_size_inches(3, 2.6)
# fig.savefig(savefile, format='pdf', dpi=300)

plt.savefig(savefile, format='pdf')
savefile.close()
