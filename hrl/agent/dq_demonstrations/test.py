from hrl.agent.dq_demonstrations.combined_replay_buffer import CombinedPrioritizedReplayBuffer

rbuf = CombinedPrioritizedReplayBuffer(
    10,
    10
)

for i in range(11):
    rbuf.append(
        'state0-{}'.format(i),
        'action0-{}'.format(i),
        i
    )
for i in range(1):
    rbuf.append(
        'state1-{}'.format(i),
        'action1-{}'.format(i),
        i,
        supervised_lambda=1
    )

while(True):
    transitions = rbuf.sample(2)
    print(transitions)
    errors = [0, 1]
    rbuf.update_errors(errors)