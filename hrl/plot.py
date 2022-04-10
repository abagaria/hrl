import os
import pickle
import argparse

import numpy as np
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

def plot_salient_event_classifier(file_path, salient_event, trajs_infos):
	preds = np.array([salient_event(info) for traj_infos in trajs_infos for info in traj_infos])
	player_xs = np.array([info['player_x'] for traj_infos in trajs_infos for info in traj_infos])
	player_ys = np.array([info['player_y'] for traj_infos in trajs_infos for info in traj_infos])

	goal_x, goal_y = salient_event.target_info['player_x'], salient_event.target_info['player_y']

	neg = plt.scatter(player_xs[preds == 0], player_ys[preds == 0], c='gold', edgecolors='k')
	pos = plt.scatter(player_xs[preds == 1], player_ys[preds == 1], c='red', edgecolors='k')
	goal = plt.scatter(goal_x, goal_y, c='green', edgecolors='k')
	print(f"=============={goal_x}_{goal_y} num_pos: ", np.sum(preds))

	plt.legend([pos, neg, goal], ['pos', 'neg', 'goal'], bbox_to_anchor=(1.05, 1))
	plt.axis('tight')
	plt.savefig(file_path)
	plt.clf()

def main():
	args = parse_args()

	experiment_dir = os.path.join(args.results_dir, args.experiment_name, "log_file_0.pkl")
	plot_learning_curve(experiment_dir)

	img_save_path = os.path.join(args.results_dir, args.experiment_name, "learning_curve.png")
	plt.savefig(img_save_path)


if __name__ == "__main__":
	main()
