import pygame
import numpy as np
import math

pygame.init() #initializing pygame
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
starting_y=track_centre_y+outer_radius_y-20

screen=pygame.display.set_mode((screen_width,screen_height))
pygame.display.set_caption("Autonomous Racing Agent")
clock=pygame.time.Clock()

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
finish_line_y=track_centre_y+outer_radius_y-20

def reset():
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
def step(action):
    global agent_x,agent_y,agent_angle,agent_speed,steps,episode_done,crashed,finished,crossed_start
    if action==0: #turn left shaprly
        agent_angle-=10
    elif action==1:
        agent_angle+=10
    elif action==2: #accelerate
        agent_speed=min(agent_speed+2,max_speed)
    elif action==3: #brake
        agent_speed=max(agent_speed-3,0)

    agent_angle=agent_angle%360
    rad=math.radians(agent_angle)
    agent_x+=agent_speed*math.sin(rad)
    agent_y-=agent_speed*math.cos(rad)

    if agent_y>finish_line_y-5:
        crossed_start=True
    if crossed_start and agent_y<finish_line_y-50:
        finished=True
    crashed=check_collision()
    step+=1
    reward=calculate_reward()
    if crashed or finished or steps>=max_steps:
        episode_done=True
    return get_state(),reward,episode_done
def check_collision():
    dx=agent_x-track_centre_x
    dy=agent_y-track_centre_y
    outer_distance=(dx/outer_radius_x)**2+(dy/outer_radius_y)**2
    inner_distance=(dx/inner_radius_x)**2+(dy/inner_radius_y)**2
    if outer_distance>1 or inner_distance<1:
        return True
    return False
def calculate_reward():
    if crashed:
        reward=-10
    elif finished:
        reward=+100
    reward=0.1
    if agent_speed<1:
        reward-=0.05
    return reward
def get_perpendicular(angle_offset=0.0):
    check_angle=agent_angle+angle_offset
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
def get_state():
    left_distance=get_perpendicular(angle_offset=-90)
    right_distance=get_perpendicular(angle_offset=90)
    left_distance=max(0,min(left_distance/100,1.0))
    right_distance=max(0,min(right_distance/100,1.0))
    speed=agent_speed/max_speed
    distance_to_finish=abs(agent_y-finish_line_y)
    distance_to_finish=max(0,min(distance_to_finish/200,1.0))
    angle=agent_angle/360.0
    return np.array([left_distance, right_distance, speed, distance_to_finish,angle],dtype=np.float32)
def render(display=True):
    screen.fill(white)
    pygame.draw.ellipse( #draw the outer wall
        screen, 
        black,
        (track_centre_x - outer_radius_x, 
         track_centre_y - outer_radius_y,
         outer_radius_x * 2,
         outer_radius_y * 2),
        5
    )
    pygame.draw.ellipse( #draw the inner wall
        screen,
        black,
        (track_centre_x - inner_radius_x,
         track_centre_y - inner_radius_y,
         inner_radius_x * 2,
         inner_radius_y * 2),
        5
    )
    pygame.draw.line( #draw the finish line
        screen,
        green,
        (track_centre_x - 40, finish_line_y),
        (track_centre_x + 40, finish_line_y),
        3
    )
    pygame.draw.circle(screen, blue, (int(agent_x), int(agent_y)), agent_size) #draw the agent
    rad = math.radians(agent_angle) #draw direction indicator
    end_x = agent_x + 15 * math.sin(rad)
    end_y = agent_y - 15 * math.cos(rad)
    pygame.draw.line(
        screen,
        red,
        (int(agent_x), int(agent_y)),
        (int(end_x), int(end_y)),
        2
    )
    font = pygame.font.Font(None, 24) #text display for spped, angle and steps
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

