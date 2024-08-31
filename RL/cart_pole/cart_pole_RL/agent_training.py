import numpy as np
import pygame
pygame.init()

from cart_pole_environment import CartPole
from agent_dqn import DQN
from dueling_PER_agent_dqn import DuelingPerDQN

action = 0
reward = 0
term = False
env = CartPole()

# agent_dqn = DQN()
agent_dqn = DuelingPerDQN()

manual_control = False

while env.is_running:
    # Start training again if the agent could not achieve a good result in 10,000 steps.
    if agent_dqn.max_step > 9000:
        env = CartPole()
        agent_dqn = DuelingPerDQN()

    # Manual control.
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
                agent_dqn.max_step -= 1000
                if agent_dqn.max_step < agent_dqn.step_counter:
                    agent_dqn.epsilon = 0
            elif event.key == pygame.K_d:
                if agent_dqn.max_step < agent_dqn.step_counter:
                    agent_dqn.max_step = agent_dqn.step_counter
                agent_dqn.max_step += 1000
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

    # Train an agent if the experience array is full.
    if agent_dqn.step_counter > agent_dqn.samples_num and agent_dqn.step_counter <= agent_dqn.max_step:
        agent_dqn.train_step()
        agent_dqn.epsilon_decrement(env.score, env.max_score)

    env.render(agent_dqn.step_counter, agent_dqn.epsilon, agent_dqn.max_step)

pygame.quit()
