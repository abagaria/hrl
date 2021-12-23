import numpy as np
from .option import ModelFreeOption
from ...salient_event.salient_event import SalientEvent


class SkillChain:
    def __init__(self, init_salient_event, target_salient_event, options, chain_id, max_num_options=5):
        self.options = options
        self.chain_id = chain_id
        self.max_num_options = max_num_options

        self.init_salient_event = init_salient_event
        self.target_salient_event = target_salient_event

        # Data structures for determining when a skill chain is completed
        self._init_descendants = []
        self._init_ancestors = []
        self._is_deemed_completed = False
        self.completing_vertex = None

    def should_continue_chaining(self):
        return not self.is_chain_completed()
    
    @staticmethod
    def edge_condition(inits):
        return np.mean(inits) > 0.7

    @staticmethod
    def should_exist_edge_between_options(my_option, other_option):
        """ Should there exist an edge from option1 -> option2? """
        assert SkillChain.is_option_type(my_option)
        assert SkillChain.is_option_type(other_option)

        if my_option.get_training_phase() == "initiation_done" and \
            other_option.get_training_phase() == "initiation_done":
            effect_set = my_option.get_effective_effect_set()
            
            if len(effect_set) > 0:
                inits = [other_option.pessimistic_is_init_true(eg.obs, eg.info) for eg in effect_set]
                is_intersecting = SkillChain.edge_condition(inits)
                return is_intersecting

        return False

    @staticmethod
    def should_exist_edge_from_event_to_option(event, option):
        """ Should there be an edge from `event` to `option`? """
        assert isinstance(event, SalientEvent)
        assert SkillChain.is_option_type(option)
        assert isinstance(option, ModelFreeOption)

        if option.get_training_phase() == "initiation_done" and option.initiation_classifier.is_initialized():
            if len(event.effect_set) > 0:  # Be careful: all([]) = True
                inits = [option.pessimistic_is_init_true(eg.obs, eg.info) for eg in event.effect_set]
                is_intersecting = SkillChain.edge_condition(inits)
                return is_intersecting

            info = {"player_x": event.get_target_position()[0],
                    "player_y": event.get_target_position()[1]}

            return option.is_init_true(event.get_target_obs(), info)

        return False

    @staticmethod
    def should_exist_edge_from_option_to_event(option, event):
        """ Should there be an edge from `option` to `event`? """
        assert isinstance(option, ModelFreeOption)
        assert isinstance(event, SalientEvent)

        if option.get_training_phase() == "initiation_done":
            effect_set = option.get_effective_effect_set()
            inits = [event(eg.info) for eg in effect_set]
            return SkillChain.edge_condition(inits)

        return False

    def should_expand_initiation_classifier(self, option):
        assert isinstance(option, ModelFreeOption), f"{type(option)}"
        assert isinstance(self.init_salient_event, SalientEvent)

        if option.initiation_classifier.is_initialized():
            if len(self.init_salient_event.effect_set) > 0:
                return any([option.pessimistic_is_init_true(eg.obs, eg.info) for eg
                            in self.init_salient_event.effect_set])
            if self.init_salient_event.get_target_position() is not None:
                obs = self.init_salient_event.get_target_obs()
                info = {"player_x": self.init_salient_event.get_target_position()[0],
                        "player_y": self.init_salient_event.get_target_position()[1]}
                return option.pessimistic_is_init_true(obs, info)
        return False

    def should_complete_chain(self, option):
        """ Check if a newly learned option completes its corresponding chain. """
        assert isinstance(option, ModelFreeOption), f"{type(option)}"

        # If there is a path from a descendant of the chain's init salient event
        # to the newly learned option, then that chain's job is done
        if self.does_descendant_complete_chain(option):
            return True

        # If not, then check if there is a direct path from the init-salient-event to the new option
        if self.should_exist_edge_from_event_to_option(self.init_salient_event, option):
            self.completing_vertex = self.init_salient_event, "init_event"
            return True

        # Finally, cap by the number of skills
        return len(self.get_trained_options()) > self.max_num_options

    def does_descendant_complete_chain(self, option):
        for descendant in self._init_descendants:
            if isinstance(descendant, SalientEvent):
                if self.should_exist_edge_from_event_to_option(descendant, option):
                    self.completing_vertex = descendant, "descendant"
                    return True
            if isinstance(descendant, ModelFreeOption):
                if self.should_exist_edge_between_options(descendant, option):
                    self.completing_vertex = descendant, "descendant"
                    return True
        return False

    def is_chain_completed(self):
        """
        The chain is considered complete when it learns an option whose initiation set covers
        at least one of the descendants of the chain's init-salient-event.

        Returns:
            is_completed (bool)
        """
        if self._is_deemed_completed:
            return True

        completed_options = self.get_trained_options()

        for option in completed_options:
            if self.should_expand_initiation_classifier(option):
                option.is_last_option = True
                if self.should_complete_chain(option):
                    self._is_deemed_completed = True
                    return True

        for option in completed_options:
            if self.should_complete_chain(option):
                option.is_last_option = True
                self._is_deemed_completed = True
                return True

        return False

    def set_init_descendants(self, descendants):
        """ The `descendants` are the set of vertices that you can get to from the chain's init-salient-event. """
        if len(descendants) > 0:
            assert all([self.is_node_type(node) for node in descendants]), f"{descendants}"
            self._init_descendants = descendants

    def set_init_ancestors(self, ancestors):
        """ The `ancestors` are the set of vertices from which you can get to the chain's init-salient-event. """
        if len(ancestors) > 0:
            assert all([self.is_node_type(node) for node in ancestors]), f"{ancestors}"
            self._init_ancestors = ancestors

    def get_trained_options(self):
        return [o for o in self.options if o.get_training_phase() == "initiation_done"]

    @staticmethod
    def is_option_type(x):
        return isinstance(x, ModelFreeOption)

    @staticmethod
    def is_node_type(x):
        return isinstance(x, (ModelFreeOption, SalientEvent))

    def __eq__(self, other):
        return self.chain_id == other.chain_id

    def __str__(self):
        return "SkillChain-{}".format(self.chain_id)

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.options)

    def __getitem__(self, item):
        return self.options[item]

    def set_chain_completed(self):
        self._is_deemed_completed = True
    
    def get_leaf_nodes_from_skill_chain(self):
        return [option for option in self.options if len(option.children) == 0]

    def get_root_nodes_from_skill_chain(self):
        return [option for option in self.options if option.parent is None]

    