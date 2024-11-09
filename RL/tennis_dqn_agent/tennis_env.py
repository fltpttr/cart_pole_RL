from numpy.random import randint
import pygame
pygame.init()


class TennisEnv:
    def __init__(self, w=400, h=300, fps=30):
        # General settings.
        self.w = w
        self.h = h
        self.sc = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Tennis DQN')
        self.font = pygame.font.SysFont('arial', 16)
        self.clock = pygame.time.Clock()
        self.fps = fps

        # Platforms.
        self.platform_1_surf = pygame.Surface((self.w // 8, 30))
        self.platform_1_rect = self.platform_1_surf.get_rect()
        self.platform_1_rect.center = (self.w // 2, self.h - 15)
        self.move_left_1 = False
        self.move_right_1 = False
        self.platform_1_speed = 18
        self.prev_platform_1_x = self.w // 2

        self.platform_2_surf = pygame.Surface((self.w // 8, 30))
        self.platform_2_rect = self.platform_2_surf.get_rect()
        self.platform_2_rect.center = (self.w // 2, 15)
        self.move_left_2 = False
        self.move_right_2 = False
        self.platform_2_speed = 18
        self.prev_platform_2_x = self.w // 2

        self.wins_1 = 0
        self.wins_2 = 0
        self.score_1 = 0
        self.max_score_1 = 0
        self.score_2 = 0
        self.max_score_2 = 0

        # Ball.
        self.ball_x = randint(20, self.w - 20)
        self.ball_y = 40
        self.prev_ball_x = self.ball_x
        self.prev_ball_y = self.ball_y
        self.ball_speed_x = 12
        self.ball_speed_y = -12
        self.ball_radius = 16

        # Expirience data.
        self.term = False
        self.prev_term = False
        self.reward_1 = 0
        self.reward_2 = 0

    def platform_1_move(self):
        self.prev_platform_1_x = self.platform_1_rect.centerx
        if self.move_left_1 and self.platform_1_rect.left > 0:
            self.platform_1_rect.centerx -= self.platform_1_speed
        elif self.move_right_1 and self.platform_1_rect.right < self.w:
            self.platform_1_rect.centerx += self.platform_1_speed

    def platform_2_move(self):
        self.prev_platform_2_x = self.platform_2_rect.centerx
        if self.move_left_2 and self.platform_2_rect.left > 0:
            self.platform_2_rect.centerx -= self.platform_2_speed
        elif self.move_right_2 and self.platform_2_rect.right < self.w:
            self.platform_2_rect.centerx += self.platform_2_speed

    def ball_move(self):
        self.prev_term = self.term
        self.term = False
        self.prev_ball_x = self.ball_x
        self.prev_ball_y = self.ball_y

        if self.ball_x - self.ball_radius < 0 or self.ball_x + self.ball_radius > self.w:
            self.ball_speed_x = -self.ball_speed_x
            if self.ball_speed_x > 0:
                self.ball_x += self.ball_speed_x + randint(6, 12)
            else:
                self.ball_x += self.ball_speed_x - randint(6, 12)

        if self.ball_y < 0 and (self.ball_x + self.ball_radius < self.platform_2_rect.left or \
             self.ball_x - self.ball_radius > self.platform_2_rect.right):
            self.reward_2 = -1
            if self.score_2 > self.max_score_2:
                self.max_score_2 = self.score_2
                self.score_2 = 0
            self.wins_1 += 1
            self.ball_x = randint(20, self.w - 20)
            self.ball_y = self.h // 2
        elif self.ball_y > self.h and (self.ball_x + self.ball_radius < self.platform_1_rect.left or \
             self.ball_x - self.ball_radius > self.platform_1_rect.right):
            self.reward_1 = -1
            if self.score_1 > self.max_score_1:
                self.max_score_1 = self.score_1
                self.score_1 = 0
            self.wins_2 += 1
            self.ball_x = randint(20, self.w - 20)
            self.ball_y = self.h // 2

        self.ball_x += self.ball_speed_x
        self.ball_y += self.ball_speed_y

    def platform_collision(self):
        if self.ball_x + self.ball_radius > self.platform_1_rect.left and \
                self.ball_x - self.ball_radius < self.platform_1_rect.right and \
                self.ball_y + self.ball_radius//2 > self.platform_1_rect.top:
            self.reward_1 = 1
            self.score_1 += 1
            self.ball_speed_y = -self.ball_speed_y
            self.ball_y += self.ball_speed_y - randint(3, 6)
        elif self.ball_x + self.ball_radius > self.platform_2_rect.left and \
                self.ball_x - self.ball_radius < self.platform_2_rect.right and \
                self.ball_y - self.ball_radius//2 < self.platform_2_rect.bottom:
            self.reward_2 = 1
            self.score_2 += 1
            self.ball_speed_y = -self.ball_speed_y
            self.ball_y += self.ball_speed_y + randint(3, 6)

    def to_give_state(self):
        if self.ball_y > self.h or self.ball_y < 0:
            self.term = True

        state_1 = [(self.platform_1_rect.centerx - self.w / 2) / self.w * 2,
                   (self.platform_1_rect.centerx - self.prev_platform_1_x) / self.platform_1_speed,
                   (self.ball_x - self.w / 2) / self.w * 2,
                   (self.ball_y - self.h / 2) / self.h * 2,
                   (self.ball_x - self.prev_ball_x) / abs(self.ball_speed_x),
                   (self.ball_y - self.prev_ball_y) / abs(self.ball_speed_y)]

        state_2 = [(self.platform_2_rect.centerx - self.w / 2) / self.w * 2,
                   (self.platform_2_rect.centerx - self.prev_platform_2_x) / self.platform_2_speed,
                   state_1[2], state_1[3], state_1[4], state_1[5]]

        if self.prev_term:
            state_1[4] = 0
            state_1[5] = 0
            state_2[4] = 0
            state_2[5] = 0

        return (state_1, self.term), (state_2, self.term)

    def to_give_reward(self):
        return self.reward_1, self.reward_2

    def reset_reward(self):
        self.reward_1 = 0
        self.reward_2 = 0

    def render(self, step_counter, epsilon, beta):
        self.sc.fill((255, 255, 255))

        pygame.draw.circle(self.sc, (255, 0, 0), (self.ball_x, self.ball_y), self.ball_radius)
        self.sc.blit(self.platform_1_surf, self.platform_1_rect)
        self.sc.blit(self.platform_2_surf, self.platform_2_rect)
        self.sc.blit(self.font.render('Step counter: ' + str(step_counter), True, (0, 0, 255)), (12, 10))
        self.sc.blit(self.font.render('Epsilon: ' + str(epsilon)[:5], True, (0, 255, 0)), (12, 30))
        pygame.draw.line(self.sc, (0, 0, 0), (8, 54), (160, 54), 1)
        self.sc.blit(self.font.render('max_score_1: ' + str(self.max_score_1), True, (255, 0, 0)), (12, 58))
        self.sc.blit(self.font.render('max_score_2: ' + str(self.max_score_2), True, (255, 0, 0)), (12, 80))
        self.sc.blit(self.font.render('beta: ' + str(beta)[:5], True, (255, 0, 0)), (12, 100))
        self.sc.blit(self.font.render('n_wins_1: ' + str(self.wins_1), True, (0, 0, 0)), (self.w - 120, self.h // 2 + 3))
        pygame.draw.line(self.sc, (0, 0, 0), (self.w, self.h // 2), (self.w - 125, self.h // 2), 1)
        self.sc.blit(self.font.render('n_wins_2: ' + str(self.wins_2), True, (0, 0, 0)), (self.w - 120, self.h // 2 - 23))

        pygame.display.update()
        self.clock.tick(self.fps)
