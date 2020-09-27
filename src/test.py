import matplotlib.pyplot as plt
import numpy as np
import os
lrfile = './LR_iter10.log'
fdmlfile = './mini_iter10.log'
lrloss =[]
fdmlloss =[]

for line in open(lrfile):
	loss = float(line.strip('\n').split(' ')[-1])
	lrloss.append(loss)
lrloss = np.array(lrloss)

for line in open(fdmlfile):
	loss = float(line.strip('\n').split(' ')[-1])
	fdmlloss.append(loss)
fdmlloss = np.array(fdmlloss)
# print(np.arange(0,len(lrloss)*10, 10))
minlen = min([len(lrloss), len(fdmlloss)])
fig = plt.figure(figsize=(10,10))
lrline = plt.plot(np.arange(0, minlen*10, 200), lrloss[:minlen:20], 'r-', label='lr loss')
fdmlline = plt.plot(np.arange(0, minlen*10, 200), fdmlloss[:minlen:20], 'b-', label='fdml loss')
# plt.legend([lrline, fdmlline], ['lr', 'fdml'])
plt.legend()
plt.axis([0, minlen, 0.6, 0.80])
fig.savefig('loss.png')