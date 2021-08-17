import os
import pickle
import argparse

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
	# plot
	plt.figure()
	plt.plot(time_steps, success_rates, 'o-')
	plt.title('learning curve')
	plt.xlabel('time step')
	plt.ylabel('success')
	plt.show()


def main(experiment_name=None, log_file_name='log_file_0.pkl', results_dir='results'):
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


if __name__ == "__main__":
	main()
