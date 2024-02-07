import os
import pickle
import argparse
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt


def parse_args():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--experiment_name', type=str, default='test',
						help='a subdirectory name for the saved results')
	parser.add_argument("--results_dir", type=str, default='results',
                        help='the name of the directory used to store results')
	args = parser.parse_args()
	return args


def plot_learning_curve(file_path):
	"""
	make success curve against timestep curve
	"""
	# get data from pickle file
	if file_path.endswith('pkl'):
		# open logging file
		with open(file_path, 'rb') as f:
			logged_data = pickle.load(f)
		# load data
		time_steps = []
		success_rates = []
		for step in logged_data:
			step_data = logged_data[step]
			if 'success' in step_data:
				time_steps.append(step)
				success_rates.append(step_data['success'])
	# get data from csv file
	elif file_path.endswith('csv'):
		df = pd.read_csv(file_path)
		time_steps = df['episode_idx']
		all_success_rates = df.loc[:, df.columns != 'episode_idx']  # many success rates for each timestep
		success_rates = all_success_rates.mean(axis=1)  # one avg success rate for each timestep
	else:
		raise NotImplementedError('only support logging learning curve in csv for pkl')
	# plot
	plt.figure()
	plt.plot(time_steps, success_rates, 'o-')
	plt.title(f'learning curve for {Path(file_path).parent.name}')
	plt.xlabel('time step')
	plt.ylabel('success rate')
	plt.show()


def main(experiment_name=None, log_file_name='testing_success_rates.csv', results_dir='results'):
	"""
	the single argument is designed solely for the purpose of calling this function
	is __main__.py
	"""
	if experiment_name is None:
		# this is used when main() in ran directly from command line
		args = parse_args()
		experiment_name = args.experiment_name
		results_dir = args.results_dir

	experiment_dir = os.path.join(results_dir, experiment_name, log_file_name)
	plot_learning_curve(experiment_dir)

	img_save_path = os.path.join(results_dir, experiment_name, "learning_curve.png")
	plt.savefig(img_save_path)
	plt.close()


if __name__ == "__main__":
	main()
