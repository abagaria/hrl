import numpy as np
from treelib import Tree
from pfrl.wrappers import atari_wrappers


class TrainingExample:
    def __init__(self, obs, info):
        assert isinstance(obs, atari_wrappers.LazyFrames)

        self.obs = obs
        self.info = self._construct_info(info)

    def _construct_info(self, x):
        return x if isinstance(x, dict) else dict(player_x=x[0], player_y=x[1])

    @property
    def pos(self):
        pos = self.info['player_x'], self.info['player_y']
        return np.array(pos)

    def __iter__(self):
        """ Allows us to iterate over an object of this class. """
        return ((self.obs, self.info) for _ in [0])


class SkillTree(object):
    def __init__(self, options):
        self._tree = Tree()
        self.options = options

        if len(options) > 0:
            [self.add_node(option) for option in options]

    def add_node(self, option):
        if option.name not in self._tree:
            print(f"Adding {option} to the skill-tree")
            self.options.append(option)
            parent = option.parent.name if option.parent is not None else None
            self._tree.create_node(tag=option.name, identifier=option.name, data=option, parent=parent)

    def get_option(self, option_name):
        if option_name in self._tree.nodes:
            node = self._tree.nodes[option_name]
            return node.data

    def get_depth(self, option):
        return self._tree.depth(option.name)

    def get_children(self, option):
        return self._tree.children(option.name)

    def traverse(self):
        """ Breadth first search traversal of the skill-tree. """
        return list(self._tree.expand_tree(mode=self._tree.WIDTH))

    def show(self):
        """ Visualize the graph by printing it to the terminal. """
        self._tree.show()