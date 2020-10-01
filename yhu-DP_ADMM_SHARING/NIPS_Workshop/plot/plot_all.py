from plot import *
import os
import numpy as np
import pickle

def load_sgd_result(result_dir, data_set):
	foo = {}
	base_dir = os.path.join(result_dir, "sgd_result/result", data_set+"_fdml_lr_0")
	train_metric_path = os.path.join(base_dir, "lr_worker_0_train_metric_by_epoch")
	foo["train_metric"] = np.genfromtxt(train_metric_path, delimiter=',')
	metric_by_check_path = os.path.join(base_dir, "lr_worker_0_metric_by_chkpt")
	foo["metric_by_check"] = np.genfromtxt(metric_by_check_path, delimiter=',')
	return foo

def load_admm_result(result_dir,data_set, noise_level="0.0", epsilon=None, delta=None):
	if None == epsilon:
		result_path = os.path.join(result_dir, "admm_result/result", data_set+"_noise_"+noise_level)
	else:
		result_path = os.path.join(result_dir, "admm_result/result", data_set+"_epsilon_"+epsilon+"_delta_"+delta)
	with open(result_path, "rb") as fin:
		history = pickle.load(fin, encoding="latin1")
	return history

def plot_a9a_loss_vs_epoch(admm, sgd, figure_dir):
	num_points_per_epoch = 10
	max_epoch = 100
	x = [range(1, max_epoch+1) for i in range(4)]
	y = []
	y.append(admm["train_objective_no_noise"][0:max_epoch])
	y.append(admm["test_logloss_no_noise"][0:max_epoch])
	y.append(sgd["train_metric"][0:max_epoch, 1])
	y.append(sgd["metric_by_check"][0:max_epoch*num_points_per_epoch:num_points_per_epoch, 2])
	line_names = ["ADMM Train Logloss+Regularizer", "ADMM Test Logloss", "SGD Train Logloss+Regularizer", "SGD Test Logloss"]
	line_styles = ["-", "--", "-", "--"]
	line_colors = ["k", "k", "b", "b"]
	line_markers = [None, None, None, None]
	line_widths = [2,2,2,2]
	path_figure = os.path.join(figure_dir, "a9a_loss_vs_epoch.eps")
	xlim=[0, max_epoch]
	ylim=[0.32, 0.4]
	xticks = [10, 20, 30, 40, 50, 60, 70, 80, 90]
	x_label = "Epoch"
	y_label = "Loss"
	figsize=(14,6)
	plot_plain_curves(x, y, path_figure, line_names, line_colors, line_styles, line_widths, line_markers, x_label, y_label, xlim, ylim, figsize, xticks)

def plot_a9a_loss_vs_time(admm, sgd, figure_dir):
	num_points_per_epoch = 10
	x = []
	y = []
	admm_time_agg = _get_agg_time_by_epoch(admm["train_time"], 1)
	sgd_time_agg = _get_agg_time_by_epoch(sgd["metric_by_check"][:, 3], num_points_per_epoch)

	x.append(admm_time_agg)
	x.append(admm_time_agg)
	x.append(sgd_time_agg)
	x.append(sgd_time_agg)
	y.append(admm["train_objective_no_noise"])
	y.append(admm["test_logloss_no_noise"])
	y.append(sgd["train_metric"][:, 1])
	y.append(sgd["metric_by_check"][0:-1:num_points_per_epoch, 2])
	sgd_len = len(sgd["train_metric"][:, 1])
	x[2] = x[2][0:sgd_len]
	x[3] = x[3][0:sgd_len]
	y[3] = y[3][0:sgd_len]

	line_names = ["ADMM Train Logloss+Regularizer", "ADMM Test Logloss", "SGD Train Logloss+Regularizer", "SGD Test Logloss"]
	line_styles = ["-", "--", "-", "--"]
	line_colors = ["k", "k", "b", "b"]
	line_markers = [None, None, None, None]
	line_widths = [2,2,2,2]
	path_figure = os.path.join(figure_dir, "a9a_loss_vs_time.eps")
	xlim=[0, 80]
	# xlim=None
	ylim=[0.32, 0.4]
	xticks =None
	x_label = "Time(s)"
	y_label = "Loss"
	figsize=(12,6)
	plot_plain_curves(x, y, path_figure, line_names, line_colors, line_styles, line_widths, line_markers, x_label, y_label, xlim, ylim, figsize, xticks)

def _get_agg_time_by_epoch(time_sequence, step):
	len_ori = len(time_sequence)
	len_res = len_ori / step
	len_record = len_res * step
	record_seq = time_sequence[0:len_record]
	agg_seq = [sum(record_seq[i*step:(i+1)*step]) for i in range(len_res)]
	res = agg_seq
	for i in range(1, len_res):
		res[i] += res[i-1]
	return res


def plot_gisette_loss_vs_epoch(admm, sgd, figure_dir):
	num_points_per_epoch = 10
	max_epoch = 100
	x = [range(1, max_epoch+1) for i in range(4)]
	y = []
	y.append(admm["train_objective_no_noise"][0:max_epoch])
	y.append(admm["test_logloss_no_noise"][0:max_epoch])
	y.append(sgd["train_metric"][0:max_epoch, 1])
	y.append(sgd["metric_by_check"][0:max_epoch*num_points_per_epoch:num_points_per_epoch, 2])
	line_names = ["ADMM Train Logloss\n+Regularizer", "ADMM Test Logloss", "SGD Train Logloss\n+Regularizer", "SGD Test Logloss"]
	line_styles = ["-", "--", "-", "--"]
	line_colors = ["k", "k", "b", "b"]
	line_markers = [None, None, None, None]
	line_widths = [2,2,2,2]
	path_figure = os.path.join(figure_dir, "gisette_loss_vs_epoch.eps")
	xlim=[0, 100]
	ylim=[0, 1]
	xticks = [i*10 for i in range(100/10)]
	x_label = "Epoch"
	y_label = "Loss"
	figsize=(12,6)
	plot_plain_curves(x, y, path_figure, line_names, line_colors, line_styles, line_widths, line_markers, x_label, y_label, xlim, ylim, figsize, xticks)


def plot_gisette_loss_vs_time(admm, sgd, figure_dir):
	num_points_per_epoch = 10
	x = []
	y = []
	admm_time_agg = _get_agg_time_by_epoch(admm["train_time"], 1)
	sgd_time_agg = _get_agg_time_by_epoch(sgd["metric_by_check"][:, 3], num_points_per_epoch)

	x.append(admm_time_agg)
	x.append(admm_time_agg)
	x.append(sgd_time_agg)
	x.append(sgd_time_agg)
	y.append(admm["train_objective_no_noise"])
	y.append(admm["test_logloss_no_noise"])
	y.append(sgd["train_metric"][:, 1])
	y.append(sgd["metric_by_check"][0:-1:num_points_per_epoch, 2])
	sgd_len = len(sgd["train_metric"][:, 1])
	x[2] = x[2][0:sgd_len]
	x[3] = x[3][0:sgd_len]
	y[3] = y[3][0:sgd_len]

	line_names = ["ADMM Train Logloss\n+Regularizer", "ADMM Test Logloss", "SGD Train Logloss\n+Regularizer", "SGD Test Logloss"]
	line_styles = ["-", "--", "-", "--"]
	line_colors = ["k", "k", "b", "b"]
	line_markers = [None, None, None, None]
	line_widths = [2,2,2,2]
	path_figure = os.path.join(figure_dir, "gisette_loss_vs_time.eps")
	xlim=[0, 160]
	# xlim=None
	ylim=[0, 1]
	xticks =None
	x_label = "Time(s)"
	y_label = "Loss"
	figsize=(12,6)
	plot_plain_curves(x, y, path_figure, line_names, line_colors, line_styles, line_widths, line_markers, x_label, y_label, xlim, ylim, figsize, xticks)


def plot_a9a_loss_vs_epoch_with_noise(noise_level_list_str, result_dir, figure_dir):
	eval_max_epoch = 100 # for region of min
	eval_epoch = 30 # for fixed point
	# construct the data
	x = [noise_level_list_str for i in range(3)]
	num_x_points = len(noise_level_list_str)
	y = []
	full_test_loss, loc_test_loss = _parse_baseline(os.path.join(result_dir, "admm_result/result/a9a_base_line"))
	y.append([full_test_loss] * num_x_points)
	y.append([loc_test_loss] * num_x_points)
	row_y = []
	for i in noise_level_list_str:
		hist = load_admm_result(result_dir,"a9a", noise_level=i)
		# row_y.append(np.min(hist["test_logloss_no_noise"][0:eval_max_epoch]))
		row_y.append(hist["test_logloss_no_noise"][eval_max_epoch])
	y.append(row_y)
	# plot the figure
	line_names = ["Full features (centralized training)", "Local features only", "ADMM with Noise"]
	line_styles = ["--", "--", "-"]
	line_colors = ["r", "b", "k"]
	line_markers = [None, None, None]
	line_widths = [2,2,2,2]
	path_figure = os.path.join(figure_dir, "a9a_test_loss_vs_noise.eps")
	# xlim=[0, 40]
	xlim=None
	ylim=None
	xticks =None
	x_label = "Standard deviation of added Gaussian noise"
	y_label = "Loss"
	figsize=(12,6)
	plot_plain_curves(x, y, path_figure, line_names, line_colors, line_styles, line_widths, line_markers, x_label, y_label, xlim, ylim, figsize, xticks)

def _parse_baseline(path):
	with open(path, "r") as fin:
		fin.readline()
		line = fin.readline()
		line_seg = line.split(",")
		full_test_loss = float(line_seg[2])
		fin.readline()
		line = fin.readline()
		line_seg = line.split(",")
		loc_test_loss = float(line_seg[2])
	return full_test_loss, loc_test_loss

def plot_gisette_loss_vs_epoch_with_noise(noise_level_list_str, result_dir, figure_dir):
	eval_max_epoch = 100 # for region of min
	eval_epoch = 50 # for fixed point
	# construct the data
	x = [noise_level_list_str for i in range(3)]
	num_x_points = len(noise_level_list_str)
	y = []
	full_test_loss, loc_test_loss = _parse_baseline(os.path.join(result_dir, "admm_result/result/gisette_base_line"))
	y.append([full_test_loss] * num_x_points)
	y.append([loc_test_loss] * num_x_points)
	row_y = []
	for i in noise_level_list_str:
		hist = load_admm_result(result_dir,"gisette", noise_level=i)
		# row_y.append(np.min(hist["test_logloss_no_noise"][0:eval_max_epoch]))
		row_y.append(hist["test_logloss_no_noise"][eval_epoch])
	y.append(row_y)
	# plot the figure
	line_names = ["Full features (centralized training)", "Local features only", "ADMM with Noise"]
	line_styles = ["--", "--", "-"]
	line_colors = ["r", "b", "k"]
	line_markers = [None, None, None]
	line_widths = [2,2,2,2]
	path_figure = os.path.join(figure_dir, "gisette_test_loss_vs_noise.eps")
	# xlim=[0, 40]
	xlim=None
	ylim=None
	xticks =None
	x_label = "Standard deviation of added Gaussian noise"
	y_label = "Loss"
	figsize=(12,6)
	plot_plain_curves(x, y, path_figure, line_names, line_colors, line_styles, line_widths, line_markers, x_label, y_label, xlim, ylim, figsize, xticks)

def plot_a9a_loss_vs_epoch_vary_noise(noise_level_list_str, result_dir, figure_dir):
	max_epoch = 100
	# load and construct the data
	num_lines = len(noise_level_list_str)
	assert num_lines <= 6
	x = [range(1, max_epoch+1) for i in range(num_lines)]
	y = []
	for i in noise_level_list_str:
		hist = load_admm_result(result_dir,"a9a", noise_level=i)
		y.append(hist["test_logloss_no_noise"][0:max_epoch])
	# plot the figure
	line_names = ["Standard deviation "+level for level in noise_level_list_str]
	line_styles = ["-", "--", "-", "--", "-", "--"]
	line_colors = ["r", "k", "b", "r", "k", "b"]
	line_markers = [None, None, None, None, None, None]
	line_widths = [2,2,2,2,2,2]
	path_figure = os.path.join(figure_dir, "a9a_test_loss_vs_epoch_vary_noise.eps")
	xlim=[0, 50]
	# xlim=None
	ylim=[0.32, 0.4]
	xticks =None
	x_label = "Epoch"
	y_label = "Testing loss"
	figsize=(12,6)
	plot_plain_curves(x, y, path_figure, line_names, line_colors, line_styles, line_widths, line_markers, x_label, y_label, xlim, ylim, figsize, xticks)

def plot_gisette_loss_vs_epoch_vary_noise(noise_level_list_str, result_dir, figure_dir):
	max_epoch = 100
	# load and construct the data
	num_lines = len(noise_level_list_str)
	assert num_lines <= 6
	x = [range(1, max_epoch+1) for i in range(num_lines)]
	y = []
	for i in noise_level_list_str:
		hist = load_admm_result(result_dir,"gisette", noise_level=i)
		y.append(hist["test_logloss_no_noise"][0:max_epoch])
	# plot the figure
	line_names = ["Standard Deviation "+level for level in noise_level_list_str]
	line_styles = ["-", "--", "-", "--", "-", "--"]
	line_colors = ["r", "k", "b", "r", "k", "b"]
	line_markers = [None, None, None, None, None, None]
	line_widths = [2,2,2,2,2,2]
	path_figure = os.path.join(figure_dir, "gisette_test_loss_vs_epoch_vary_noise.eps")
	xlim=[0, 80]
	# xlim=None
	ylim=[0, 0.8]
	xticks =None
	x_label = "Epoch"
	y_label = "Testing loss"
	figsize=(12,6)
	plot_plain_curves(x, y, path_figure, line_names, line_colors, line_styles, line_widths, line_markers, x_label, y_label, xlim, ylim, figsize, xticks)

# def plot_a9a_loss_vs_epoch_vary_epsilon(epsilon_list_str, result_dir, figure_dir):
# 	max_epoch = 100
# 	# load and construct the data
# 	num_lines = len(noise_level_list_str)
# 	assert num_lines == 3
# 	x = [range(1, max_epoch+1) for i in range(num_lines)]
# 	y = []
# 	for i in noise_level_list_str:
# 		hist = load_admm_result(result_dir,"a9a", epsilon=i, delta="0.01")
# 		y.append(hist["test_logloss_no_noise"][0:max_epoch])
# 	# plot the figure
# 	line_names = ["$\epsilon$ "+level for level in noise_level_list_str]
# 	line_styles = ["-", "-", "-", "--", "--"]
# 	line_colors = ["r", "k", "b", "k", "k"]
# 	line_markers = [None, None, None, None, None]
# 	line_widths = [2,2,2,2,2]
# 	path_figure = os.path.join(figure_dir, "a9a_test_loss_vs_epoch_vary_epsilon.eps")
# 	xlim=[0, 50]
# 	# xlim=None
# 	ylim=[0.32, 0.4]
# 	xticks =None
# 	x_label = "Epoch"
# 	y_label = "Testing loss"
# 	figsize=(12,6)
# 	plot_plain_curves(x, y, path_figure, line_names, line_colors, line_styles, line_widths, line_markers, x_label, y_label, xlim, ylim, figsize, xticks)

if __name__=="__main__":
	figure_dir = "../figures"
	result_dir = "./result"
### for normal part
## for a9a data set
	a9a_admm_history_0_0 = load_admm_result(result_dir,"a9a", "0.0")
	a9a_sgd = load_sgd_result(result_dir, "a9a")
	plot_a9a_loss_vs_epoch(a9a_admm_history_0_0, a9a_sgd, figure_dir)
	# plot_a9a_loss_vs_time(a9a_admm_history_0_0, a9a_sgd, figure_dir)
## for gisette data set
	# gisette_admm_history_0_0 = load_admm_result(result_dir,"gisette", "0.0")
	# gisette_sgd = load_sgd_result(result_dir, "gisette")
	# plot_gisette_loss_vs_epoch(gisette_admm_history_0_0, gisette_sgd, figure_dir)
	# plot_gisette_loss_vs_time(gisette_admm_history_0_0, gisette_sgd, figure_dir)

### for dp part
## perform vs noise
	# noise_level_list_str = ["0.0", "0.03", "0.1", "0.3", "1.0", "3.0"]
	# plot_a9a_loss_vs_epoch_with_noise(noise_level_list_str, result_dir, figure_dir)
	# noise_level_list_str = ["0.0", "0.1", "0.3", "1.0", "3.0", "10.0"]
	# plot_gisette_loss_vs_epoch_with_noise(noise_level_list_str, result_dir, figure_dir)
## Training convergence vs noise
	noise_level_list_str = ["0.0", "0.3", "1.0"]
	plot_a9a_loss_vs_epoch_vary_noise(noise_level_list_str, result_dir, figure_dir)
	# noise_level_list_str = ["0.0", "1.0", "3.0"]
	# plot_gisette_loss_vs_epoch_vary_noise(noise_level_list_str, result_dir, figure_dir)
## Training convergence vs noise and baselines, noise in epsilon delta
