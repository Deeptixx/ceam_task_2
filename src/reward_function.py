def get_reward(crashed, finished, speed): #basic reward function
    if crashed:
        return -10
    elif finished:
        return 100
    else:
        reward = 0.1
    if speed < 1.0:
        reward -= 0.05
    return reward
def get_reward_fast(crashed, finished, speed): #reward fucntion that encourages faster driving
    if crashed:
        return -15
    elif finished:
        return 150
    else:
        reward = 0.2
    if speed < 2.0:
        reward -= 0.1
    return reward
def get_reward_safe(crashed, finished, speed): #reward function that encourages safer driving.
    if crashed:
        return -20
    elif finished:
        return 100
    else:
        reward = 0.15
    if speed < 0.5:
        reward -= 0.02
    return reward