import numpy as np
import os
import matplotlib.pyplot as plt
import rl_utils
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
env_name = "Hopper"

return_list = []

listdir = os.listdir("./Hopper")

for file in listdir:
    data = np.load("./Hopper/"+file,"r")
    return_list.append(data)


episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('TD3 on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('TD3 on {}'.format(env_name))
plt.show()