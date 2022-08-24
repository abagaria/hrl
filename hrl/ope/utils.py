import pickle
import gym
import numpy as np

from hrl.agent.td3.utils import load
from hrl.agent.dsc.MBOptionClass import ModelBasedOption
from hrl.wrappers.antmaze_wrapper import D4RLAntMazeWrapper
from hrl.salient_event.SalientEventClass import SalientEvent
from hrl.agent.td3.TD3AgentClass import TD3

def load_chain(base_fname):
    chain = []
    # TODO: load env from saved params rather than default
    env = gym.make('antmaze-umaze-v0')
    goal_state = np.array(env.target_goal)
    mdp = D4RLAntMazeWrapper(
        env,
        start_state=np.array((0, 0)),
        goal_state=goal_state,
        init_truncate=True,
        use_dense_reward=False)

    options_params_fname = base_fname + '_options_params.pkl'
    with open(options_params_fname, 'rb') as f:
        options_params = pickle.load(f)

    # Load global option's value learner (agent initialized using the global option's parameters
    value_learner = TD3(state_dim=mdp.state_space_size()+2,
                        action_dim=mdp.action_space_size(),
                        max_action=1.,
                        name=f"{options_params[0]['name']}-td3-agent",
                        device=options_params[0]["device"],
                        lr_c=options_params[0]["lr_c"],
                        lr_a=options_params[0]["lr_a"],
                        use_output_normalization=options_params[0]["use_model"])
    load(value_learner, base_fname + "_value_learner")

    ######################
    # Load global_option #
    ######################

    global_option_params_fname = base_fname + '_global_option_params.pkl'
    with open(global_option_params_fname, 'rb') as f:
        global_option_params = pickle.load(f)

    target_salient_event = SalientEvent(
        global_option_params['target_salient_event_data']["target_state"],
        global_option_params['target_salient_event_data']["event_idx"],
        global_option_params['target_salient_event_data']["tolerance"],
        global_option_params['target_salient_event_data']["intersection_event"],
        global_option_params['target_salient_event_data']["is_init_event"],
    )
    target_salient_event.trigger_points = global_option_params['target_salient_event_data']["trigger_points"]
    target_salient_event.revised_by_mpc = global_option_params['target_salient_event_data']["revised_by_mpc"]
    # Finished loading target_salient_event
    global_option = ModelBasedOption(
        parent=None,
        mdp=mdp,
        buffer_length=global_option_params["buffer_length"],
        global_init=global_option_params["global_init"],
        gestation_period=global_option_params["gestation_period"],
        timeout=global_option_params["timeout"],
        max_steps=global_option_params["max_steps"],
        device=global_option_params["device"],
        target_salient_event=target_salient_event,
        name=global_option_params["name"],
        # path_to_model=options_params[0]["path_to_model"],
        global_solver=None,
        use_vf=global_option_params["use_vf"],
        use_global_vf=global_option_params["use_global_vf"],
        use_model=global_option_params["use_model"],
        dense_reward=global_option_params["dense_reward"],
        global_value_learner=None,
        option_idx=0,
        lr_c=global_option_params["lr_c"],
        lr_a=global_option_params["lr_a"],
        multithread_mpc=global_option_params["multithread_mpc"],
        init_classifier_type=global_option_params["init_classifier_type"],
        optimistic_threshold=global_option_params["optimistic_threshold"],
        pessimistic_threshold=global_option_params["pessimistic_threshold"])
    global_option.value_learner = value_learner  # Hopefully I did not introduce a bug since the global option will initialize a new TD3 agent as value_learner which this line will replace

    global_option.initiation_classifier = global_option_params["initiation_classifier"]
    global_option.success_curve = global_option_params["success_curve"]
    global_option.effect_set = global_option_params["effect_set"]

    ######################################
    ##### Load and reconstruct chain #####
    ######################################

    idx = 1
    for current_option_params in options_params:
        # Load target_salient_event
        # TODO: check if SalientEvent can just be pickled directly
        target_salient_event = SalientEvent(
            current_option_params['target_salient_event_data']["target_state"],
            current_option_params['target_salient_event_data']["event_idx"],
            current_option_params['target_salient_event_data']["tolerance"],
            current_option_params['target_salient_event_data']["intersection_event"],
            current_option_params['target_salient_event_data']["is_init_event"],
        )
        target_salient_event.trigger_points = current_option_params['target_salient_event_data']["trigger_points"]
        target_salient_event.revised_by_mpc = current_option_params['target_salient_event_data']["revised_by_mpc"]
        # Finished loading target_salient_event

        assert idx == current_option_params["option_idx"]
        if "parent_idx" in current_option_params.keys():
            parent = chain[current_option_params["parent_idx"]]
        else:
            parent = None
        option = ModelBasedOption(parent=parent,
                                  mdp=mdp,
                                  buffer_length=current_option_params["buffer_length"],
                                  global_init=current_option_params["global_init"],
                                  gestation_period=current_option_params["gestation_period"],
                                  timeout=current_option_params["timeout"],
                                  max_steps=current_option_params["max_steps"],
                                  device=current_option_params["device"],
                                  target_salient_event=target_salient_event,
                                  name=current_option_params["name"],
                                  path_to_model="",
                                  global_solver=None,
                                  use_vf=current_option_params["use_vf"],
                                  use_global_vf=current_option_params["use_global_vf"],
                                  use_model=current_option_params["use_model"],
                                  dense_reward=current_option_params["dense_reward"],
                                  global_value_learner=value_learner,
                                  option_idx=current_option_params["option_idx"],
                                  lr_c=current_option_params["lr_c"],
                                  lr_a=current_option_params["lr_a"],
                                  multithread_mpc=current_option_params["multithread_mpc"],
                                  init_classifier_type=current_option_params["init_classifier_type"],
                                  optimistic_threshold=current_option_params["optimistic_threshold"],
                                  pessimistic_threshold=current_option_params["pessimistic_threshold"])

        option.initiation_classifier = current_option_params["initiation_classifier"]
        option.success_curve = current_option_params["success_curve"]
        option.effect_set = current_option_params["effect_set"]

        #TODO: Load (and save in dsc) o.children

        chain.append(option)
        idx += 1
    return global_option, chain
