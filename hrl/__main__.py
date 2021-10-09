from hrl.tasks.monte.MRRAMMDPClass import MontezumaRAMMDP

mdp = MontezumaRAMMDP(render=False, seed=0)
mdp.saveImage("img1")
for i in range(5):
    a = mdp.sample_random_action()
    mdp.step(a)
    mdp.saveImage(f"img{i+2}")
print("success!")