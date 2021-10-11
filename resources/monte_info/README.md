# saved info of monte

the `dm` agent space files indicate the agent space after the environment is
wrapped by the typical deepmind wrappers.

## middle ladder bottom
the goal state is the botton of the first ladder in monte
`middle_ladder_bottom.npy`: image of agent in the goal state
`middle_ladder_bottom_pos.txt`: x, y position of the goal state

`middle_ladder_bottom_agent_space.npy`: image of the agent's surrounding pixels in the goal state

## right ladder
the right ladder in the ladder on the right side in the first room in monte
`right_ladder_top.npy`: image of agent standing at the top of the right ladder
`right_ladder_top_pos.txt`: x, y position of the agent when standing on top of the right ladder

`right_ladder_top_agent_space.npy`: image of the agent's surrounding pixels when standing on top of right ladder

## left ladder 
the left ladder is the ladder on the left in the first room
`left_ladder_bottom.npy`: image of the agent standing at bottom of left ladder
`left_ladder_bottom_pos.txt`: x, y position of the agent when standing at bottom of left ladder
`left_ladder_top.npy`: image of the agent standing at top of left ladder
`left_ladder_top_pos.txt`: x, y position of the agent when standing at top of left ladder

`left_ladder_bottom_agent_space.npy`: image of the agent's surrounding pixels when standing at bottom of left ladder
`left_ladder_top_agent_space.npy`: image of the agent's surrounding pixels when standing at top of left ladder
