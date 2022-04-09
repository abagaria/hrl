from sklearn.metrics.pairwise import cosine_similarity

from label_extractors.label_extractor import LabelExtractor

class PositiveAugmentExtractor(LabelExtractor):
    def __init__(self, feature_extractor, extract_only_positive, window_sz=5, num_trajs=2, cos_threshold=0.95):
        '''
        Args:
            feature_extractor (FeatureExtractor): used to extract features to calc cosine similarity
            extract_only_positive (bool): if true, only return positive egs
            window_sz (int): how many states before or after to count as positive egs
            num_trajs (int): num of non-subgoal trajs to augment data with
            cos_threshold (float): cosine similarity threshold for whether a state is a positive example
        '''
        self.feature_extractor = feature_extractor
        self.extract_only_pos = extract_only_positive
        self.window_sz = window_sz
        self.num_trajs = num_trajs
        self.cos_threshold = cos_threshold

    def is_positive_eg(self, subgoal_state, state):
        subgoal_state_feats = self.feature_extractor.extract_features([subgoal_state])
        state_feats = self.feature_extractor.extract_features([state])
        return cosine_similarity(subgoal_state_feats, state_feats)[0][0] > self.cos_threshold

    def extract_labels(self, state_trajs, raw_ram_trajs, subgoal_traj_idx, subgoal_state_idx):
        '''
        Extract labels from given state trajectories and the idx of the subgoal.

        Note that the PositiveAugmentExtractor has 4 classes of labels:
            1. Positive, in subgoal trajectory (1)
            2. Negative, in subgoal trajectory (0)
            3. Negative, states outside of subgoal trajectory below cos similarity threshold (2)
            4. Positive, states outside of subgoal trajectory above cos similarity threshold (3)

        Args:
            state_traj (list (list(np.array))): state trajectories
            raw_ram_trajs (list (list(np.array))): state trajectories - RawRAM states
            subgoal_traj_idx (int): index of traj containing the subgoal
            subgoal_state_idx (int): index of chosen subgoal

        Returns:
            (list(np.array)): list of np.array of states
            (list(int)): list of labels of corresponding states
        '''
        subgoal_traj = state_trajs[subgoal_traj_idx]
        subgoal_state = subgoal_traj[subgoal_state_idx]

        pos_start = max(0, subgoal_state_idx - self.window_sz)
        pos_end = min(len(subgoal_traj), subgoal_state_idx + self.window_sz)

        pos_idxs = list(range(pos_start, pos_end + 1))
        pos_states = [subgoal_traj[i] for i in pos_idxs]

        if not self.extract_only_pos:
            subgoal_neg_idxs = [i for i in range(len(subgoal_traj)) if i < pos_start or i > pos_end]
            subgoal_neg_states = [subgoal_traj[i] for i in subgoal_neg_idxs] 

            non_subgoal_trajs_start = max(0, subgoal_traj_idx - self.num_trajs)
            non_subgoal_trajs_end = min(len(state_trajs), self.num_trajs + non_subgoal_trajs_start + 1)
            non_subgoal_trajs = [state_trajs[i] for i in range(non_subgoal_trajs_start, subgoal_traj_idx)]\
                                + [state_trajs[i] for i in range(subgoal_traj_idx + 1, non_subgoal_trajs_end)]

            non_subgoal_neg_states = []
            non_subgoal_pos_states = []
            for non_subgoal_traj in non_subgoal_trajs:
                for state in non_subgoal_traj:
                    if self.is_positive_eg(subgoal_state, state):
                        non_subgoal_pos_states.append(state)
                    else:
                        non_subgoal_neg_states.append(state)
        else:
            subgoal_neg_states = []
            non_subgoal_neg_states = []

        states = pos_states + subgoal_neg_states + non_subgoal_neg_states + non_subgoal_pos_states
        labels = [1 for _ in range(len(pos_states))] \
            + [0 for _ in range(len(subgoal_neg_states))] \
            + [2 for _ in range(len(non_subgoal_neg_states))] \
            + [3 for _ in range(len(non_subgoal_pos_states))]
        return states, labels
