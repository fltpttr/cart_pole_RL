"""Environment."""
import random
import pygame
pygame.init()

class CartPole:
    def __init__(self, w=600, h=400, fps=30, cart_acceleration=0.04):
        # General settings.
        self.w = w
        self.h = h
        self.fps = fps
        self.is_running = True
        self.is_render = True
        self.score = 0
        self.max_score = 0
        self.sc = pygame.display.set_mode((w, h))
        pygame.display.set_caption('CartPole')
        self.clock = pygame.time.Clock()
        self.acceleration_time = 0
        self.font = pygame.font.SysFont('arial', 16)

        # The cart object.
        self.__cart_surf = pygame.Surface((60, 40))
        self.__cart_surf.fill((0, 0, 0))
        self.__cart_rect = self.__cart_surf.get_rect()
        self.__cart_rect.center = (self.w // 2, self.h // 2 + 100)
        self.__cart_acceleration = cart_acceleration
        self.__cart_speed = 0
        self.move_left = False
        self.move_right = False
        self.__prev_cart_x = self.__cart_rect.centerx

        # The pole object.
        self.__pole_length = 150
        self.__pole_surf = pygame.Surface((10, self.__pole_length))
        self.__rotated_pole_surf = pygame.Surface((10, self.__pole_length))
        self.__pole_surf.fill((225, 116, 0))
        self.__angle = 185
        self.__delta_angle = 0
        self.max_angle = 40

    def move_cart(self, action):
        # action = 0 - move left, action = 1 - move right.
        if action == 0:
            if self.move_left:
                self.acceleration_time = 0
            self.move_right = True
            self.move_left = False
        elif action == 1:
            if self.move_right:
                self.acceleration_time = 0
            self.move_left = True
            self.move_right = False

        if self.move_right:
            self.acceleration_time += 1
            self.__cart_rect.centerx += int(self.__cart_acceleration * (self.acceleration_time ** 2))
        elif self.move_left:
            self.acceleration_time += 1
            self.__cart_rect.centerx -= int(self.__cart_acceleration * (self.acceleration_time ** 2))

        self.__cart_speed = self.__cart_rect.centerx - self.__prev_cart_x
        self.__prev_cart_x = self.__cart_rect.centerx

    def rotate_surf(self, pos, angle):
        pole_w, pole_h = self.__pole_surf.get_size()
        img2 = pygame.Surface((pole_w * 2, pole_h * 2), pygame.SRCALPHA)
        img2 = img2.convert_alpha()
        img2.fill((0, 0, 0, 0))
        img2.blit(self.__pole_surf, (pole_w - pos[0], pole_h - pos[1]))
        return pygame.transform.rotate(img2, angle)

    def rotate_pole(self):
        self.__delta_angle = (self.__angle - 180) / 10 + self.__cart_speed / 2
        self.__angle = self.__angle + self.__delta_angle
        self.__rotated_pole_surf = self.rotate_surf((5, 0), self.__angle)
        self.__rotated_pole_rect = self.__rotated_pole_surf.get_rect()
        self.__rotated_pole_rect.center = self.__cart_rect.center

    def give_state(self):
        return [(self.__cart_rect.centerx - self.w // 2) / self.w * 2, self.__cart_speed / 20,
                (self.__angle - 180) / self.max_angle, self.__delta_angle / 3]

    def score_increment(self):
        self.score += 1

    def lose(self):
        if self.__cart_rect.right > self.w or self.__cart_rect.left < 0 or self.__angle > 180 + self.max_angle or \
           self.__angle < 180 - self.max_angle:
            if self.score > self.max_score:
                self.max_score = self.score
            self.score = 0
            self.acceleration_time = 0
            self.__angle = random.choice([175, 185])
            self.__cart_rect.centerx = self.w // 2
            reward = -1
            term = True
        else:
            reward = 0.1
            term = False
        return reward, term

    def render(self, step_counter, epsilon):
        if self.is_render:
            self.sc.fill((255, 255, 255))
            pygame.draw.line(self.sc, (0, 0, 0), (0, 300), (self.w, 300), 1)
            self.sc.blit(self.__cart_surf, self.__cart_rect)
            self.sc.blit(self.__rotated_pole_surf, self.__rotated_pole_rect)
            pygame.draw.circle(self.sc, (0, 110, 200), self.__cart_rect.center, 8)

            self.sc.blit(self.font.render('Score: ' + str(self.score), True, (0, 0, 0)), (15, 15))
            self.sc.blit(self.font.render('Max score: ' + str(self.max_score), True, (255, 0, 0)), (15, 40))
            self.sc.blit(self.font.render('Step counter: ' + str(step_counter), True, (0, 0, 255)), (15, 65))
            self.sc.blit(self.font.render('Epsilon: ' + str(epsilon)[:5], True, (0, 255, 0)), (15, 90))

            pygame.display.update()
            self.clock.tick(self.fps)
