import pygame
pygame.init()
from tennis_env import TennisEnv
# from tennis_agent_dqn import DQN
from tennis_agent_dueling_dqn import DQN
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

isRunning = True
visualize_fl = True
manual_fl = True
env = TennisEnv()
tennis_agent_1 = DQN()
tennis_agent_2 = DQN()
action_1 = -1
action_2 = -1

while isRunning:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            isRunning = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                isRunning = False
            elif event.key == pygame.K_a:
                tennis_agent_1.epsilon -= 0.1
                tennis_agent_2.epsilon -= 0.1
            elif event.key == pygame.K_d:
                tennis_agent_1.epsilon = 1
                tennis_agent_2.epsilon = 1
            elif event.key == pygame.K_v:
                visualize_fl = not visualize_fl
            elif event.key == pygame.K_m:
                manual_fl = not manual_fl
            elif event.key == pygame.K_LEFT:
                action_1 = 0
            elif event.key == pygame.K_RIGHT:
                action_1 = 1
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                env.move_left_1 = False
            elif event.key == pygame.K_RIGHT:
                env.move_right_1 = False

    state_1, state_2 = env.to_give_state()

    if not manual_fl:
        action_1 = tennis_agent_1.choose_action(state_1[0])
    action_2 = tennis_agent_2.choose_action(state_2[0])

    if action_1 == 0:
        env.move_right_1 = False
        env.move_left_1 = True
    elif action_1 == 1:
        env.move_left_1 = False
        env.move_right_1 = True
    if action_2 == 0:
        env.move_right_2 = False
        env.move_left_2 = True
    elif action_2 == 1:
        env.move_left_2 = False
        env.move_right_2 = True

    tennis_agent_1.save_state(*state_1)
    tennis_agent_2.save_state(*state_2)
    env.platform_1_move()
    env.platform_2_move()
    env.ball_move()
    env.platform_collision()
    tennis_agent_1.save_reward(env.to_give_reward()[0])
    tennis_agent_2.save_reward(env.to_give_reward()[1])
    env.reset_reward()
    if not manual_fl:
        tennis_agent_1.train_model()
    tennis_agent_1.step_counter_increment()
    tennis_agent_2.train_model()
    tennis_agent_2.step_counter_increment()

    if visualize_fl:
        env.render(tennis_agent_1.step_counter, tennis_agent_1.epsilon, tennis_agent_1.beta)

    if manual_fl:
        action_1 = -1

# tennis_agent_1.save_model('agent_ddqn_1_1')
# tennis_agent_2.save_model('agent_ddqn_1_2')
pygame.quit()
