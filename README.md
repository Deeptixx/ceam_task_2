# Autonomous Racing agent using reinforcement learning

## Project Overview
This project implements an autonomous racing agent capable of learning how to navigate a custom 2D racing track using reinforcement learning.
The given task was to design and train an agent that begines with no prior knowledge of the environment and gradually learns to successfully complete a lap through interaction with the environment and rewards. Unlike supervised learning here the agent learns through trial and error. The agent repeatedly observes the environment, performs actions, recieves rewardsand keeps updating its policy.
This eas implemented using Stable-Baselines3 with the proximal policy optimization algorithm and a multi layer perception(MLP) policy.

## Project Structure
Project Structure
ceam_task_2/

├── models/
│   └── trained_racing_agent.zip
│
├── results/
│   ├── analysis/
│
├── src/
│   ├── environment.py
│   ├── reward_function.py
│   ├── agent.py 
│
├── training_evaluation/
│   ├── train.py
│   ├── test.py
│   ├── visualize.py
│
└── README.md
## Environment Design
The environment consists of:
- An outer boundary representing the edge of the track
- An inner boundary represeting the inner wall
- A finish line
- Collision detection
- Episode termination conditions
The objective of the agent is to complete a full lap while avoiding collisions and crossing the finish line whilst ending close enough to the staring position so that it is a considered a full lap (agent starts about 100 pixels away)
Each episode terminate when:
- The agent crashes
- The agent finishes a lap
- Max episode length reached

## Learning Process and results
The agent initially had no knowlegde of the track and selected actions randomly because of which it often stayed at the same place or crashed immediately or timed out before completing the lap
As training progressed the reward function was refined, the agent learnt to move more consistently, although it struggled initially to reach the finish line further imporvements to the reward function led to stable behaviours
The final racing agent succesfully navigated the track. The ep_rew_mean went from -24 to +630 consequently the ep_len_mean also increased (with occasional dips)

### State information ( observation space)
The agent recieves numerical observations describing its surroundings
The observation vector contains seven values
- left_far (dist to left wall)
- left_mid ( dist to left front wall)
- front (dist ahead)
- right_far (dist to right wall)
- right_mid (dist to right front wall)
- speed
- heading angle (current orientation)
Distnace sensors allow the agent to estimate nearby onstacles, speed tells the agent about current motion, and angle, the agents orientation.
### Action space
4 possible actions (turn right, turn left, brake, accelerate)
## Reward Design
- Crash:-50 , discourages collision with the track boundaries
- Finish:+500 , strongly rewards finishing a lap
- Base reward = 0.1
- Movement Reward: (movement=math.sqrt((agent_x-prev_agent_x)** 2 + (agent_y-prev_agent_y) ** 2))*0.05
encourages agent to keep moving rather than remaining stationary
- Speed Reward: speed*0.02 ,  for forward progress

## Falied Behaviour
- **Agent remained stationary**
  - Initially the reward function did not really encourage movement, the agent learnt that if it moves it will crash which causes penalty which is too risky so it stayed at the same place to maximize rewards , so I introduced a small movement reward so that the agent keeps moving and exploring the environment 
- **Agent timed out**
  - Frequenty exceed the episode length so I tweaked the reward function a little bit with speed* 0.2 forward progress , I also tried increasing the movement reward initially it was movement_2 which was too high and the agent kept crashing or once again episode kept timing out, this speed*0.2 seemed to work
- **Agent creahed near the finish line at the inner wall**
  - After learning most of the track the agent kept crashing near the finish line at the inner wall so i introduced a center penalty

## Improvements made
- **Finish line detection**
  - Initially my finish line logic was a little flawed because I applied a logic that assumed the track was a straight path and my agent never got around to learning but after refining it and taking into account the elliptical track the agent could successfully complete laps
- Evaluation plots
 - Used to measure success rate, crash rate, timeout rate, average reward and average number of steps (to analyse the final policy)

## Design decisions
- **Why PPO and not DQN?**
  - I chose PPO for this task because it provides more stable policy updates which is better for sequential decision making tasks like like racing agents. DQN learns Q-values and requires an experience replay buffer and target network while PPO directly learns policy making training simpler.
- **Why MLP instead of CNN?**
   - The agent recieves numerical sensor reading rather than image inputs, MLP is simpler and more appropriate than a convolutional neural networks 
- **Why custom environment?**
  - This was just to provide deeper understanding of reinforcement learning enevironments

## Conclusion 
Through this project, I gained a much better understanding of how reinforcement learning works in practice. Starting with an agent that had no knowledge of the track, I improved its behaviour by refining the reward function, observations, environment through several iterations until it was able to complete the track successfully.
Overall, this project helped me understand the importance of reward design, observation selection, and continuous experimentation in reinforcement learning. It also gave me hands-on experience in building, training, debugging, and evaluating an autonomous racing agent.
