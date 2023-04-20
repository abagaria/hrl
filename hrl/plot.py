import glob
import pickle
import argparse
import numpy as np

from matplotlib import pyplot as plt


def get_scores(data_dir):
  """Return a list of lists of success rates."""
  success_curves = []
  for log_file in glob.glob(f"{data_dir}/log_file_*.pkl"):
    log = pickle.load(open(log_file, 'rb'))
    curve = [log[episode]['success'] for episode in log if 'success' in log[episode]]
    success_curves.append(curve)
  return success_curves


def truncate(scores, max_length=-1, min_length=-1):
  filtered_scores = [score_list for score_list in scores if len(score_list) > min_length]
  if not filtered_scores:
    return filtered_scores
  lens = [len(x) for x in filtered_scores]
  print('lens: ', lens)
  min_length = min(lens)
  if max_length > 0:
    min_length = min(min_length, max_length)
  truncated_scores = [score[:min_length] for score in filtered_scores]
  
  return truncated_scores


def get_plot_params(array):
  median = np.median(array, axis=0)
  means = np.mean(array, axis=0)
  std = np.std(array, axis=0)
  N = array.shape[0]
  top = means + (std / np.sqrt(N))
  bot = means - (std / np.sqrt(N))
  return median, means, top, bot


def moving_average(a, n=25):
  ret = np.cumsum(a, dtype=float)
  ret[n:] = ret[n:] - ret[:-n]
  return ret[n-1:] / n


def smoothen_data(scores, n=10):
  smoothened_cols = scores.shape[1] - n + 1
  smoothened_data = np.zeros((scores.shape[0], smoothened_cols))
  for i in range(scores.shape[0]):
    smoothened_data[i, :] = moving_average(scores[i, :], n=n)
  return smoothened_data


def generate_plot(score_array, label, smoothen=0, linewidth=2, all_seeds=False):
  # smoothen is a number of iterations to average over
  if smoothen > 0:
    score_array = smoothen_data(score_array, n=smoothen)
  median, mean, top, bottom = get_plot_params(score_array)
  plt.plot(mean, linewidth=linewidth, label=label)
  plt.fill_between( range(len(top)), top, bottom, alpha=0.2 )
  if all_seeds:
    print(score_array.shape)
    for i, score in enumerate(score_array):
      plt.plot(score, linewidth=linewidth, label=label+f"_{i+1}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--experiment_name', type=str, default='test',
            help='a subdirectory name for the saved results')
  parser.add_argument("--results_dir", type=str, default='results',
                        help='the name of the directory used to store results')
  args = parser.parse_args()

  # experiments = ['unweighted_binary_antumaze_g10',
  #                'weighted_binary_antumaze_g10',]
  
  # experiments = ['unweighted_binary_ant_medium_maze_g10_reachability',
  #                'weighted_binary_ant_medium_maze_g10_reachability',]
  # experiments = ['neg_weighted_ant_medium_maze_reachability_goal_sampling_g5',
  #                'neg_weighted_ant_medium_maze_reachability_goal_sampling_g10']
  experiments = ['negweight_reachability_sum_goal_sampling_medium_maze_g5_s0biased',
                 'negweight_reachability_sum_goal_sampling_medium_maze_g10_s0biased',
                 'negweight_first_goal_sampling_medium_maze_g5_s0biased',
                 'posneg_weighted_reachability_goal_sampling_medium_maze_g5_s0biased',
                 'vanilla_binary_reachability_softmax_goal_sampling_medium_maze_g5_s0biased']
  
  plt.figure(figsize=(20, 12))
  
  for experiment in experiments:
    print(experiment)
    data_dir = f'results/{experiment}'
    score_list = get_scores(data_dir)
    scores = truncate(score_list)
    score_array = np.array(scores)
    generate_plot(score_array, experiment, smoothen=20, all_seeds=True)
  
  plt.legend()
  plt.savefig('neg_weighted_ant_medium_maze_reachability_goal_sampling.png')
  plt.close()
