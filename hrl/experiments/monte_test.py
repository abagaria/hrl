from hrl.tasks.monte.MRRAMMDPClass import MontezumaRAMMDP

mdp = MontezumaRAMMDP(render=False, seed=0)
mdp.saveImage("img1")
mdp.set_player_position(21, 192)
mdp.saveImage("img2")
print("success!")

# python -m hrl.experiments.monte_test