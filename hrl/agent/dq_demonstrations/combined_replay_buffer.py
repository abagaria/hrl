from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar
import collections
from pfrl.replay_buffers import PrioritizedReplayBuffer
from pfrl.collections.prioritized import TreeQueue, PrioritizedBuffer
from pfrl.utils.random import sample_n_k
import numpy as np

# tuple<value, index, tree>
NodeInfo = Tuple[float, int, int]
Node = List[Any]
T = TypeVar("T")
V = TypeVar("V")
Aggregator = Callable[[Sequence[V]], V]

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

def _reduce(node: Node, op: Aggregator):
    assert node
    left_node, right_node, _ = node
    parent_value = []
    if left_node:
        parent_value.append(left_node[2][0])
    if right_node:
        parent_value.append(right_node[2][0])
    if parent_value:
        node[2] = (op(parent_value), node[2][1], node[2][2])
    else:
        del node[:]

def _expand(node: Node) -> None:
    if not node:
        node[:] = [], [], (0.0, [], [])

def _write(
    index_left: int,
    index_right: int,
    node: Node,
    key: int,
    value: Optional[V],
    op: Aggregator,
) -> Optional[V]:
    if index_right - index_left == 1:
        if node:
            ret = node[2]
        else:
            ret = None
        if value is None:
            del node[:]
        else:
            node[:] = None, None, value
    else:
        _expand(node)
        node_left, node_right, _ = node
        index_center = (index_left + index_right) // 2
        if key < index_center:
            ret = _write(index_left, index_center, node_left, key, value, op)
        else:
            ret = _write(index_center, index_right, node_right, key, value, op)
        _reduce(node, op)
    return ret

class CombinedSumTreeQueue(TreeQueue[NodeInfo]):

    def __init__(self):
        super().__init__(op=sum)

    def _write(self, ix: int, val: Optional[V]) -> Optional[V]:
        ixl, ixr = self.bounds
        return _write(ixl, ixr, self.root, ix, val, self.op)

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
    
    def _write(self, ix: int, val: Optional[V]) -> Optional[V]:
        ixl, ixr = self.bounds
        return _write(ixl, ixr, self.root, ix, val, self.op)

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
        # check if experience and if experience buffer is full
        if not is_demonstration and (self.capacity is not None and self.experience_buffer_size() == self.capacity):
            self.popleft(remove_experience=True)
        # check if demonstration and if demonstration buffer is full
        if is_demonstration and (self.demonstration_capacity is not None and self.demonstration_buffer_size() == self.demonstration_capacity):
            self.popleft(remove_experience=False)

        if priority is None:
            # Append with the highest priority
            priority = self.max_priority
        if is_demonstration:
            priority += self.demonstration_priority_bonus
        else:
            priority += self.experience_priority_bonus
        if is_demonstration:
            tree_index = 1
            buffer_index = self.demonstration_idx
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
            print(pop)
            self.priority_sums.append(pop)
        for pop in popped_min:
            print(pop)
            self.priority_mins.append(pop)
        if remove_experience:
            self.experience_front = (self.experience_front + 1) % self.capacity
            return self.data[0].popleft()
        else:
            self.demonstration_front = (self.demonstration_front + 1) % self.demonstration_capacity
            return self.data[1].popleft()
        

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

    def _sample_indices_and_probabilities(self, 
            n: int, uniform_ratio: float
    ) -> Tuple[List[int], List[float], float]:
        total_priority: float = self.priority_sums.sum()
        min_prob = self.priority_mins.min() / total_priority
        indices = []
        priorities = []
        if uniform_ratio > 0:
            # Mix uniform samples and prioritized samples
            n_uniform = np.random.binomial(n, uniform_ratio)
            un_indices, un_priorities = self.priority_sums.uniform_sample(
                n_uniform, remove=self.wait_priority_after_sampling
            )
            indices.extend(un_indices)
            priorities.extend(un_priorities)
            n -= n_uniform
            min_prob = uniform_ratio/(self.experience_buffer_size() + self.demonstration_buffer_size()) + (1 - uniform_ratio)*min_prob

        pr_indices, pr_priorities = self.priority_sums.prioritized_sample(
            n, remove=self.wait_priority_after_sampling
        )
        indices.extend(pr_indices)
        priorities.extend(pr_priorities)

        print(priorities)

        probs = []
        for pri in priorities:
            prob = uniform_ratio / (self.experience_buffer_size() + self.demonstration_buffer_size()) + (1 - uniform_ratio) * pri[0]/total_priority
            probs.append((prob, pri[1], pri[2]))

        return indices, probs, min_prob

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
            





