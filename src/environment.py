import pygame
import numpy as np
import math
from reward_function import get_reward, get_reward_fast, get_reward_safe
pygame.init() #initlaize pygame
screen_width=600
screen_height=400
fps=60
white=(255,255,255)
black=(0,0,0)
green=(0,255,0)
red=(255,0,0)
blue=(0,0,255)
yellow=(255,255,0)
track_centre_x=screen_width/2
track_centre_y=screen_height/2
track_width=80
outer_radius_x=150
outer_radius_y=100
inner_radius_x=outer_radius_x-track_width
inner_radius_y=outer_radius_y-track_width
agent_size=8
max_speed=10
starting_x=track_centre_x
starting_y=track_centre_y+outer_radius_y-30

screen=pygame.display.set_mode((screen_width,screen_height)) #set up display and clock for controlling frame rate
pygame.display.set_caption("Autonomous Racing Agent")
clock=pygame.time.Clock()

agent_x=starting_x  #inital state of the agent
agent_y=starting_y
agent_angle=0
agent_speed=0
steps=0
max_steps=1000
episode_done=False
crashed=False
finished=False
crossed_start=False
finish_line_x=track_centre_x

def reset(): #reset the environment to the initial state and return the inital observtions
    global agent_x,agent_y,agent_angle,agent_speed,steps,episode_done,crashed,finished,crossed_start
    agent_x=starting_x
    agent_y=starting_y
    agent_angle=0
    agent_speed=0
    steps=0
    max_steps=1000
    episode_done=False
    crashed=False
    finished=False
    crossed_start=False
    return get_state()

def step(action): #update the environment based on action and return new state and reward
    global agent_x,agent_y,agent_angle,agent_speed,steps,episode_done,crashed,finished,crossed_start
    if action==0:
        agent_angle-=3
    elif action==1:
        agent_angle+=3
    elif action==2:
        agent_speed=min(agent_speed+2,max_speed)
    elif action==3:
        agent_speed=max(agent_speed-2,0)
    agent_angle=agent_angle%360
    rad=math.radians(agent_angle)
    agent_x+=agent_speed*math.sin(rad)
    agent_y-=agent_speed*math.cos(rad)
    if agent_x>finish_line_x-5 and agent_x<finish_line_x+5:
        crossed_start=True
    if crossed_start and (agent_x>finish_line_x+20 or agent_x<finish_line_x-20):
        finished=True
    crashed=check_collision()
    steps+=1
    reward=get_reward(crashed, finished, agent_speed)
    
    if crashed or finished or steps>=max_steps:
        episode_done=True
    return get_state(),reward,episode_done

def check_collision(): #check if the agent collides with the track boundaries
    dx=agent_x-track_centre_x
    dy=agent_y-track_centre_y
    outer_distance=(dx/outer_radius_x)**2+(dy/outer_radius_y)**2 #calculate distance from the track centre and normalize by the track radii
    inner_distance=(dx/inner_radius_x)**2+(dy/inner_radius_y)**2
    if outer_distance>1 or inner_distance<1:
        return True
    return False

def get_perpendicular(angle_offset=0.0): #cast a ray in the direction of the agent's angle plus the offset and check for collision with track boundaries to measure distance to the wall
    check_angle=agent_angle+angle_offset #angle offset is used to check left and right and simulate many sensor directions 
    rad=math.radians(check_angle)
    for distance in range(1,150):
        check_x=agent_x+distance*math.sin(rad)
        check_y=agent_y-distance*math.cos(rad)
        dx=check_x-track_centre_x
        dy=check_y-track_centre_y
        outer_distance=(dx/outer_radius_x)**2+(dy/outer_radius_y)**2
        inner_distance=(dx/inner_radius_x)**2+(dy/inner_radius_y)**2
        if outer_distance>1.0 or inner_distance<1.0:
            return distance
    return 150

def get_state(): #returns current state in the form of a numpy array(normalized)
    left_distance=get_perpendicular(angle_offset=-90)
    right_distance=get_perpendicular(angle_offset=90)
    left_distance=max(0,min(left_distance/100,1.0))
    right_distance=max(0,min(right_distance/100,1.0))
    speed=agent_speed/max_speed
    distance_to_finish=abs(agent_x-finish_line_x)
    distance_to_finish=max(0,min(distance_to_finish/200,1.0))
    angle=agent_angle/360.0
    return np.array([left_distance, right_distance, speed, distance_to_finish,angle],dtype=np.float32)

def render(display=True): # draw track
    screen.fill(white)
    pygame.draw.ellipse(
        screen, 
        black,
        (track_centre_x - outer_radius_x, 
         track_centre_y - outer_radius_y,
         outer_radius_x * 2,
         outer_radius_y * 2),
        5
    )
    pygame.draw.ellipse(
        screen,
        black,
        (track_centre_x - inner_radius_x,
         track_centre_y - inner_radius_y,
         inner_radius_x * 2,
         inner_radius_y * 2),
        5
    )
    pygame.draw.line(
        screen,
        green,
        (track_centre_x,track_centre_y+inner_radius_y),
        (track_centre_x, track_centre_y+outer_radius_y),
        3
    )
    pygame.draw.circle(screen, blue, (int(agent_x), int(agent_y)), agent_size)
    rad = math.radians(agent_angle)
    tip_x = agent_x + 15 * math.sin(rad)
    tip_y = agent_y - 15 * math.cos(rad)
    left_x = agent_x + 8 * math.sin(rad + math.radians(140))
    left_y = agent_y - 8 * math.cos(rad + math.radians(140))
    right_x = agent_x + 8 * math.sin(rad - math.radians(140))
    right_y = agent_y - 8 * math.cos(rad - math.radians(140))
    pygame.draw.polygon(
        screen,
        red,
        [(int(tip_x), int(tip_y)), (int(left_x), int(left_y)), (int(right_x), int(right_y))]
    )
    font = pygame.font.Font(None, 24)
    speed_text = font.render(f"speed: {agent_speed:.1f}", True, black)
    angle_text = font.render(f"angle: {agent_angle:.0f}°", True, black)
    step_text = font.render(f"steps: {steps}", True, black)
    screen.blit(speed_text, (10, 10))
    screen.blit(angle_text, (10, 40))
    screen.blit(step_text, (10, 70))
    
    if crashed:
        crash_text = font.render("CRASHED", True, red)
        screen.blit(crash_text, (screen_width - 150, 10))
    
    if finished:
        finish_text = font.render("FINISHED", True, green)
        screen.blit(finish_text, (screen_width - 180, 10))
    pygame.display.flip()
    clock.tick(fps)

if __name__ == "__main__": #test
    state = reset()
    
    print("Testing Racing Environment")
    print("Arrow keys: LEFT/RIGHT to turn, UP to accelerate, DOWN to brake")
    print("Q to quit")
    print(f"Initial state: {state}")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        
        keys = pygame.key.get_pressed()
        action = None
        
        if keys[pygame.K_LEFT]:
            action = 0
        elif keys[pygame.K_RIGHT]:
            action = 1
        elif keys[pygame.K_UP]:
            action = 2
        elif keys[pygame.K_DOWN]:
            action = 3
        else:
            action = 0
        
        state, reward, done = step(action)
        render()
        
        if done:
            print(f"Episode done. Crashed: {crashed}, Finished: {finished}, Total steps: {steps}")
            state = reset()
    
    print("Test complete!")
    pygame.quit()