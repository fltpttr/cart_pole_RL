import pygame
pygame.init()
from tennis_env import TennisEnv
from tennis_agent_dqn import DQN
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

isRunning = True
visualize_fl = True
env = TennisEnv()
tennis_agent = DQN()

while isRunning:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            isRunning = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                isRunning = False
            elif event.key == pygame.K_a:
                tennis_agent.epsilon -= 0.1
            elif event.key == pygame.K_d:
                tennis_agent.epsilon += 0.1
            elif event.key == pygame.K_v:
                visualize_fl = not visualize_fl
            elif event.key == pygame.K_LEFT:
                env.move_right_1 = False
                env.move_left_1 = True
            elif event.key == pygame.K_RIGHT:
                env.move_left_1 = False
                env.move_right_1 = True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                env.move_left_1 = False
            elif event.key == pygame.K_RIGHT:
                env.move_right_1 = False

    state = env.to_give_state()
    action = tennis_agent.choose_action(state[0])

    if action == 0:
        env.move_right_1 = False
        env.move_left_1 = True
    elif action == 1:
        env.move_left_1 = False
        env.move_right_1 = True

    tennis_agent.save_state(*state)
    env.platform_move()
    env.ball_move()
    env.platform_collision()
    tennis_agent.save_reward(env.to_give_reward())
    env.reset_reward()
    tennis_agent.train_model()
    tennis_agent.step_counter_increment()
    if visualize_fl:
        env.render(tennis_agent.step_counter, tennis_agent.epsilon)

pygame.quit()
