def get_reward(crashed, finished, speed):
    if crashed:
        return -50
    if finished:
        return 500
    
    reward = 0.1
    reward += speed * 0.2  # Increased from 1.5
    return reward