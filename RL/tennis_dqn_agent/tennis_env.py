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
        self.score = 0
        self.max_score = 0
        self.font = pygame.font.SysFont('arial', 16)
        self.clock = pygame.time.Clock()
        self.fps = fps

        # Platforms.
        self.platform_1_surf = pygame.Surface((self.w // 5, 30))
        self.platform_1_rect = self.platform_1_surf.get_rect()
        self.platform_1_rect.center = (self.w // 2, self.h - 15)
        self.move_left_1 = False
        self.move_right_1 = False
        self.platform_1_speed = 18
        self.prev_platform_1_x = self.w // 2

        # Ball.
        self.ball_x = randint(20, self.w - 20)
        self.ball_y = 40
        self.prev_ball_x = self.ball_x
        self.prev_ball_y = self.ball_y
        self.ball_speed_x = 20
        self.ball_speed_y = -15
        self.ball_radius = 18
        self.ball_collision = False

        # Expirience data.
        self.term = False
        self.prev_term = False
        self.reward = 0

    def platform_move(self):
        self.prev_platform_1_x = self.platform_1_rect.centerx
        if self.move_left_1 and self.platform_1_rect.left > 0:
            self.platform_1_rect.centerx -= self.platform_1_speed
        elif self.move_right_1 and self.platform_1_rect.right < self.w:
            self.platform_1_rect.centerx += self.platform_1_speed

    def ball_move(self):
        self.prev_term = self.term
        self.term = False
        self.prev_ball_x = self.ball_x
        self.prev_ball_y = self.ball_y

        if self.ball_x < 0 or self.ball_x > self.w:
            self.ball_speed_x = -self.ball_speed_x
            if self.ball_speed_x > 0:
                self.ball_x += self.ball_speed_x + randint(6, 12)
            else:
                self.ball_x += self.ball_speed_x - randint(6, 12)
        if self.ball_y < 0:
            self.ball_collision = False
            self.ball_speed_y = -self.ball_speed_y
            if self.ball_speed_y > 0:
                self.ball_y += self.ball_speed_y + randint(6, 12)
            else:
                self.ball_y += self.ball_speed_y - randint(6, 12)

        # Lose.
        elif self.ball_y > self.h:
            self.ball_collision = False
            if self.score > self.max_score:
                self.max_score = self.score
            self.score = 0
            self.reward = -1
            self.ball_x = randint(20, self.w - 20)
            self.ball_y = 40

        self.ball_x += self.ball_speed_x
        self.ball_y += self.ball_speed_y

    def platform_collision(self):
        if self.ball_x > self.platform_1_rect.left and self.ball_x < self.platform_1_rect.right and \
                self.ball_y > self.platform_1_rect.top and not self.ball_collision:
            self.ball_collision = True
            self.score += 1
            self.reward = 1
            self.ball_speed_y = -self.ball_speed_y
            if self.ball_speed_y > 0:
                self.ball_y += self.ball_speed_y + randint(6, 12) * 4
            else:
                self.ball_y += self.ball_speed_y - randint(6, 12) * 4

    def to_give_state(self):
        if self.ball_y > self.h:
            self.term = True

        state = [(self.platform_1_rect.centerx - self.w / 2) / self.w * 2,
                 (self.platform_1_rect.centerx - self.prev_platform_1_x) / self.platform_1_speed,
                 (self.ball_x - self.w / 2) / self.w * 2,
                 (self.ball_y - self.h / 2) / self.h * 2,
                 (self.ball_x - self.prev_ball_x) / abs(self.ball_speed_x),
                 (self.ball_y - self.prev_ball_y) / abs(self.ball_speed_y)]

        if self.prev_term:
            state[4] = 0
            state[5] = 0

        return state, self.term

    def to_give_reward(self):
        return self.reward

    def reset_reward(self):
        self.reward = 0

    def render(self, step_counter, epsilon):
        self.sc.fill((255, 255, 255))

        pygame.draw.circle(self.sc, (255, 0, 0), (self.ball_x, self.ball_y), self.ball_radius)
        self.sc.blit(self.platform_1_surf, self.platform_1_rect)
        self.sc.blit(self.font.render('Score: ' + str(self.score), True, (0, 0, 0)), (12, 12))
        self.sc.blit(self.font.render('Max score: ' + str(self.max_score), True, (255, 0, 0)), (12, 30))
        self.sc.blit(self.font.render('Step counter: ' + str(step_counter), True, (0, 0, 255)), (12, 50))
        self.sc.blit(self.font.render('Epsilon: ' + str(epsilon)[:5], True, (0, 255, 0)), (12, 70))

        pygame.display.update()
        self.clock.tick(self.fps)
