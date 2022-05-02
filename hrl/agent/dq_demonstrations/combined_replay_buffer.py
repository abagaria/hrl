from typing import List, Optional, Sequence, Tuple, TypeVar
import collections
from pfrl.replay_buffers import PrioritizedReplayBuffer
from pfrl.collections.prioritized import TreeQueue, PrioritizedBuffer
from pfrl.utils.random import sample_n_k
import numpy as np

# tuple<value, index, tree>
NodeInfo = Tuple[float, int, int]
T = TypeVar("T")

def _find(index_left: int, index_right: int, node: None, pos: float) -> int:
    if index_right - index_left == 1:
        return index_left
    else:
        node_left, node_right, _ = node
        index_center = (index_left + index_right) // 2
        if node_left: 
            left_value = node_left[2][0]
        else:
            left_value = 0.0
        if pos < left_value:
            return _find(index_left, index_center, node_left, pos)
        else:
            return _find(index_center, index_right, node_right, pos - left_value)

class CombinedSumTreeQueue(TreeQueue[NodeInfo]):

    def __init__(self):
        super().__init__(op=sum)

    def sum(self):
        if self.length == 0:
            return 0.0
        else:
            return self.root[2][0]

    def uniform_sample(self, n: int, remove: bool) -> Tuple[List[int], List[NodeInfo]]:
        assert n >= 0
        ixs = list(sample_n_k(self.length, n))
        vals: List[NodeInfo] = []
        if n > 0:
            for ix in ixs:
                val = self._write(ix, (0.0, -1, -1))
                assert val is not None
                vals.append(val)

        if not remove:
            for ix, val in zip(ixs, vals):
                self._write(ix, val)
        
        return ixs, vals

    def prioritized_sample(self, n: int, remove: bool) -> Tuple[List[int], List[NodeInfo]]:
        assert n >= 0
        ixs: List[int] = []
        vals: List[NodeInfo] = []
        if n > 0:
            root = self.root
            ixl, ixr = self.bounds
            for _ in range(n):
                ix = _find(ixl, ixr, root, np.random.uniform(0.0, root[2][0]))
                val = self._write(ix, (0.0, -1, -1))
                assert val is not None
                ixs.append(ix)
                vals.append(val)

        if not remove:
            for ix, val in zip(ixs, vals):
                self._write(ix, val)

        return ixs, vals

class CombinedMinTreeQueue(TreeQueue[NodeInfo]):
    def __init__(self):
        super().__init__(op=min)

    def min(self) -> float:
        if self.length == 0:
            return np.inf
        else:
            return self.root[2][0]


class CombinedPriorityBuffer(PrioritizedBuffer):

    def __init__(self, 
            capacity: Optional[int] = None, 
            demonstration_capacity: Optional[int] = None,
            wait_priority_after_sampling: bool = True, 
            initial_max_priority: float = 1,
            experience_priority_bonus=0.001,
            demonstration_priority_bonus=1.0):
        
        super().__init__(capacity, 
                wait_priority_after_sampling, 
                initial_max_priority)

        self.demonstration_capacity = demonstration_capacity
        if demonstration_capacity is None and capacity is not None:
            self.demonstration_capacity = capacity

        self.experience_priority_bonus = experience_priority_bonus
        self.demonstration_priority_bonus = demonstration_priority_bonus

        self.priority_sums = CombinedSumTreeQueue()
        self.priority_mins = CombinedMinTreeQueue()

        # data[0]: experience replay
        # data[1]: demonstration trajectories
        self.data = [
            collections.deque(),
            collections.deque()
        ]

        self.experience_idx = 0
        self.experience_front = 0
        self.demonstration_idx = 0
        self.demonstration_front = 0

    def __len__(self) -> int:
        #return length of experience buffer
        return len(self.data[0])

    def demonstration_buffer_size(self):
        return len(self.data[1])

    def experience_buffer_size(self):
        return len(self.data[0])

    def append(self, value: T, priority: Optional[float] = None, is_demonstration = False):
        if self.capacity is not None and len(self) == self.capacity:
            self.popleft()
        if priority is None:
            # Append with the highest priority
            priority = self.max_priority
        if is_demonstration:
            priority += self.demonstration_priority_bonus
        else:
            priority += self.experience_priority_bonus
        if is_demonstration:
            tree_index = 1
            buffer_index = self.expert_idx
        else:
            tree_index = 0
            buffer_index = self.experience_idx

        self.data[tree_index].append(value)
        self.priority_sums.append(
            (priority, buffer_index, tree_index)
        )
        self.priority_mins.append(
            (priority, buffer_index, tree_index)
        )

        if is_demonstration:
            self.demonstration_idx = (self.demonstration_idx + 1) % self.demonstration_capacity
        else:
            self.experience_idx = (self.experience_idx + 1) % self.capacity

    def popleft(self, remove_experience) -> T:
        assert len(self) > 0
        if remove_experience:
            tree_index = 0
        else:
            tree_index = 1
        popped_sum = []
        popped_min = []
        found = False
        while not found:
            sum_pop = self.priority_sums.popleft()
            min_pop = self.priority_mins.popleft()
            if sum_pop[2] == tree_index:
                found = True
            else:
                popped_sum.append(sum_pop)
                popped_min.append(min_pop)
        for pop in popped_sum:
            self.priority_sums.append(pop)
        for pop in popped_min:
            self.priority_mins.append(pop)
        if remove_experience:
            self.experience_front = (self.experience_idx + 1) % self.capacity
        else:
            self.expert
        return self.data[0].popleft()

    def _get_true_index(self, index, tree_index):
        if tree_index == 1:
            return (self.demonstration_capacity - self.demonstration_front + index) % self.demonstration_capacity
        else:
            return (self.capacity - self.experience_front + index) % self.capacity

    def sample(self, 
        n: int, uniform_ratio: float = 0) -> Tuple[List[T], List[float], float]:
        assert not self.wait_priority_after_sampling or not self.flag_wait_priority
        indices, node_info, min_prob = self._sample_indices_and_probabilities(
            n, uniform_ratio=uniform_ratio
        )
        self.sampled_indices = indices
        self.flag_wait_priority = True

        probabilities = [node_info[i][0] for i in range(len(node_info))]
        true_indices = [self._get_true_index(node_info[i][1], node_info[i][0]) for i in range(len(node_info))]
        tree_indices = [node_info[i][2] for i in range(len(node_info))]

        sampled = []

        for i in range(len(node_info)):
            sampled.append(self.data[tree_indices[i]][true_indices[i]])

        return sampled, probabilities, min_prob

class CombinedPrioritizedReplayBuffer(PrioritizedReplayBuffer):

    def __init__(self, 
            capacity=None, 
            demonstration_capacity=None,
            alpha=0.6, 
            beta0=0.4, 
            betasteps=200000, 
            eps=0.01, 
            normalize_by_max=True, 
            error_min=0, 
            error_max=1, 
            num_steps=1):
        super().__init__(capacity, alpha, beta0, betasteps, eps, normalize_by_max, error_min, error_max, num_steps)

        self.demonstration_capacity = demonstration_capacity
        self.memory = CombinedPriorityBuffer(capacity=capacity, demonstration_capacity=demonstration_capacity)

    def append(self, 
            state, 
            action, 
            reward, 
            next_state=None, 
            next_action=None, 
            is_state_terminal=False, 
            env_id=0,
            supervised_lambda=0,
            **kwargs):
        
        last_n_transitions = self.last_n_transitions[env_id]
        experience = dict(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            next_action=next_action,
            is_state_terminal=is_state_terminal,
            supervised_lambda=supervised_lambda,
            **kwargs
        )
        is_demonstration = (supervised_lambda != 0)
        last_n_transitions.append(experience)
        if is_state_terminal:
            while last_n_transitions:
                self.memory.append(list(last_n_transitions), is_demonstration=is_demonstration)
                del last_n_transitions[0]
            assert len(last_n_transitions) == 0
        else:
            if len(last_n_transitions) == self.num_steps:
                self.memory.append(list(last_n_transitions), is_demonstration=is_demonstration)
            





