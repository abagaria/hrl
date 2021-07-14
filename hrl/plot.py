import os
import pickle
import argparse

from matplotlib import pyplot as plt


def parse_args():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--experiment_name', type=str, default='test',
						help='a subdirectory name for the saved results')
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


def main():
	args = parse_args()

	experiment_dir = os.path.join(args.experiment_name, "log_file_0.pkl")
	plot_learning_curve(experiment_dir)

	img_save_path = os.path.join(args.experiment_name, "learning_curve.png")
	plt.savefig(img_save_path)


if __name__ == "__main__":
	main()
