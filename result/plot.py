import matplotlib.pyplot as plt
import pickle
import matplotlib
import tkinter
import os.path

matplotlib.use("TkAgg")
path = os.path.join(os.path.dirname(__file__), 'a9a_noise_0.3')
with open(path, 'rb') as f:
	pklf = pickle.load(f)
traindata = pklf['train_objective_no_noise']
testdata = pklf['test_logloss_no_noise']
data = [traindata, testdata]
#ylim=[0.32, 0.4]
ax = plt.subplot(1,1,1)
ax.plot(range(0,100), traindata)
ax.plot(range(0,100), testdata)
#ax.set_ylim(ylim)
plt.show()