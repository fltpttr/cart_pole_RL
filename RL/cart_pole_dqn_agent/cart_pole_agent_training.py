import numpy as np
import pygame
pygame.init()

from cart_pole_environment import CartPole
from cart_pole_agent_dqn import DQN
from cart_pole_dueling_PER_agent_dqn import DuelingPerDQN

action = 0
reward = 0
term = False
env = CartPole()

# agent_dqn = DQN(epsilon_step=0.0001)
agent_dqn = DuelingPerDQN(epsilon_step=0.00015, per=True)

manual_control = False

while env.is_running:
    for event in pygame.event.get():
        # Quit.
        if event.type == pygame.QUIT:
            env.is_running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                env.move_right = True
                env.move_left = False
                env.acceleration_time = 0
            elif event.key == pygame.K_LEFT:
                env.move_left = True
                env.move_right = False
                env.acceleration_time = 0
            # Quit - key 'q'.
            elif event.key == pygame.K_q:
                env.is_running = False
            # Switch to manual control.
            elif event.key == pygame.K_m:
                manual_control = not manual_control
                env.move_right = False
                env.move_left = False
                env.acceleration_time = 0
                break
            # Epsilon manual control - keys 'a' and 'd'.
            elif event.key == pygame.K_a:
                agent_dqn.epsilon -= 0.1
            elif event.key == pygame.K_d:
                agent_dqn.epsilon += 0.1
            # Render off/on - key 'v'.
            elif event.key == pygame.K_v:
                env.is_render = not env.is_render
            # Restart episode - key 'r'.
            elif event.key == pygame.K_r:
                env.__angle = np.random.randint(185 - env.max_angle, 175 + env.max_angle)
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_RIGHT:
                env.move_right = False
                env.acceleration_time = 0
            elif event.key == pygame.K_LEFT:
                env.move_left = False
                env.acceleration_time = 0

    state_sample = env.give_state()
    agent_dqn.save_state(state_sample, term)
    action = agent_dqn.choosing_action()

    if manual_control:
        action = -1

    env.move_cart(action)
    env.rotate_pole()
    env.score_increment()
    reward, term = env.lose()
    agent_dqn.save_reward(reward)
    agent_dqn.step_increment()
    agent_dqn.train_step()

    env.render(agent_dqn.step_counter, agent_dqn.epsilon)

pygame.quit()
