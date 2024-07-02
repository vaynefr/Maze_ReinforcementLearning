import pygame
import numpy as np
import random
import sys
import time


# Function to draw text
def draw_text(text, font, color, surface, x, y):
    text_obj = font.render(text, True, color)
    text_rect = text_obj.get_rect()
    text_rect.center = (x, y)
    surface.blit(text_obj, text_rect)


# Function to draw rounded rectangle
def draw_rounded_rect(surface, rect, color, radius):
    pygame.draw.rect(surface, color, rect)
    circle_radius = radius
    circle_diameter = radius * 2
    top_left_circle_center = rect.topleft
    top_right_circle_center = rect.topright
    bottom_left_circle_center = rect.bottomleft
    bottom_right_circle_center = rect.bottomright


# Hide and Seek Environment
class HideAndSeekEnv:
    def __init__(self, grid_size=12):
        self.grid_size = grid_size
        self.walls = [
            (1, 1),
            (1, 2),
            (1, 4),
            (1, 5),
            (1, 7),
            (1, 8),
            (1, 10),
            (2, 4),
            (2, 7),
            (2, 10),
            (3, 1),
            (3, 2),
            (3, 4),
            (3, 7),
            (3, 8),
            (3, 10),
            (4, 1),
            (4, 7),
            (4, 10),
            (5, 1),
            (5, 4),
            (5, 5),
            (5, 7),
            (5, 10),
            (6, 5),
            (7, 1),
            (7, 4),
            (7, 7),
            (7, 8),
            (7, 10),
            (8, 1),
            (8, 4),
            (8, 10),
            (9, 1),
            (9, 4),
            (9, 5),
            (9, 7),
            (9, 8),
            (9, 10),
            (10, 1),
            (10, 7),
            (10, 10),
            (11, 1),
            (11, 2),
            (11, 4),
            (11, 5),
            (11, 7),
            (11, 8),
            (11, 10),
        ]
        self.end = (11, 11)
        self.reset()

    def reset(self):
        while True:
            self.hider_pos = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            )
            self.seeker_pos = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            )
            if (
                self.hider_pos != self.seeker_pos
                and self.hider_pos not in self.walls
                and self.seeker_pos not in self.walls
            ):
                break
        return self._get_obs()

    def _get_obs(self):
        return np.array([*self.hider_pos, *self.seeker_pos])

    def step(self, hider_action, seeker_action):
        self.hider_pos = self._move_agent(hider_action, self.hider_pos)
        self.seeker_pos = self._move_agent(seeker_action, self.seeker_pos)
        reward_hider, reward_seeker, done = self._compute_reward()
        return self._get_obs(), reward_hider, reward_seeker, done

    def _move_agent(self, action, pos):
        new_pos = list(pos)
        if action == 0 and pos[0] > 0:
            new_pos[0] -= 1  # up
        elif action == 1 and pos[0] < self.grid_size - 1:
            new_pos[0] += 1  # down
        elif action == 2 and pos[1] > 0:
            new_pos[1] -= 1  # left
        elif action == 3 and pos[1] < self.grid_size - 1:
            new_pos[1] += 1  # right

        if tuple(new_pos) not in self.walls:
            return tuple(new_pos)
        return pos

    def _compute_reward(self):
        if self.hider_pos == self.seeker_pos:
            return -10, 10, True  # seeker wins
        elif self.hider_pos == self.end:
            return 10, -10, True  # hider wins by reaching the end
        else:
            return (
                -0.1,
                -0.1,
                False,
            )  # small penalty for each move to encourage faster solutions


class QLearningAgent:
    def __init__(self, grid_size=12, lr=0.1, gamma=0.95, epsilon=0.1):
        self.grid_size = grid_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((grid_size, grid_size, 4))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(4)
        return np.argmax(self.q_table[state[0], state[1]])

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state[0], state[1], action]
        target = reward + self.gamma * np.max(
            self.q_table[next_state[0], next_state[1]]
        )
        self.q_table[state[0], state[1], action] += self.lr * (target - predict)

    def save(self, filename):
        np.save(filename, self.q_table)

    def load(self, filename):
        self.q_table = np.load(filename)


def visualize(env, hider_agent, seeker_agent, episodes=10, max_steps_per_episode=100):
    pygame.init()
    cell_size = 50
    screen_size = env.grid_size * cell_size
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("Hide and Seek")

    font = pygame.font.Font(None, 36)

    def draw_grid():
        for x in range(0, screen_size, cell_size):
            for y in range(0, screen_size, cell_size):
                rect = pygame.Rect(x, y, cell_size, cell_size)
                pygame.draw.rect(screen, (200, 200, 200), rect, 1)
        for wall in env.walls:
            rect = pygame.Rect(
                wall[1] * cell_size, wall[0] * cell_size, cell_size, cell_size
            )
            pygame.draw.rect(screen, (0, 0, 0), rect)

    for episode in range(episodes):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps_per_episode:
            screen.fill((255, 255, 255))
            draw_grid()

            hider_action = np.argmax(hider_agent.q_table[state[0], state[1]])
            seeker_action = np.argmax(seeker_agent.q_table[state[2], state[3]])
            next_state, reward_hider, reward_seeker, done = env.step(
                hider_action, seeker_action
            )
            state = next_state
            steps += 1

            hider_pos = (
                state[1] * cell_size + cell_size // 2,
                state[0] * cell_size + cell_size // 2,
            )
            seeker_pos = (
                state[3] * cell_size + cell_size // 2,
                state[2] * cell_size + cell_size // 2,
            )

            pygame.draw.circle(screen, (0, 0, 255), hider_pos, cell_size // 3)
            pygame.draw.circle(screen, (255, 0, 0), seeker_pos, cell_size // 3)

            # Draw goal by filling the grid cell with green color
            goal_pos = (env.end[1] * cell_size, env.end[0] * cell_size)
            pygame.draw.rect(
                screen, (0, 255, 0), (goal_pos[0], goal_pos[1], cell_size, cell_size)
            )

            hider_label = font.render("H", True, (0, 0, 0))
            seeker_label = font.render("S", True, (0, 0, 0))

            screen.blit(
                hider_label,
                (
                    hider_pos[0] - hider_label.get_width() // 2,
                    hider_pos[1] - hider_label.get_height() // 2,
                ),
            )
            screen.blit(
                seeker_label,
                (
                    seeker_pos[0] - seeker_label.get_width() // 2,
                    seeker_pos[1] - seeker_label.get_height() // 2,
                ),
            )

            pygame.display.flip()
            pygame.time.delay(300)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    pygame.quit()
                    return

    pygame.quit()


def rl_agent():
    import numpy as np
    import pygame
    import random
    import time

    class HideAndSeekEnv:
        def __init__(self, grid_size=12):
            self.grid_size = grid_size
            self.walls = [
                (1, 1),
                (1, 2),
                (1, 4),
                (1, 5),
                (1, 7),
                (1, 8),
                (1, 10),
                (2, 4),
                (2, 7),
                (2, 10),
                (3, 1),
                (3, 2),
                (3, 4),
                (3, 7),
                (3, 8),
                (3, 10),
                (4, 1),
                (4, 7),
                (4, 10),
                (5, 1),
                (5, 4),
                (5, 5),
                (5, 7),
                (5, 10),
                (6, 5),
                (6, 10),
                (7, 1),
                (7, 4),
                (7, 7),
                (7, 8),
                (7, 10),
                (8, 1),
                (8, 4),
                (8, 10),
                (9, 1),
                (9, 4),
                (9, 5),
                (9, 7),
                (9, 8),
                (9, 10),
                (10, 1),
                (10, 7),
                (10, 10),
                (11, 1),
                (11, 2),
                (11, 4),
                (11, 5),
                (11, 7),
                (11, 8),
                (11, 10),
            ]
            self.end = (11, 11)
            self.reset()

        def reset(self):
            while True:
                self.hider_pos = (
                    random.randint(0, self.grid_size - 1),
                    random.randint(0, self.grid_size - 1),
                )
                self.seeker_pos = (
                    random.randint(0, self.grid_size - 1),
                    random.randint(0, self.grid_size - 1),
                )
                if (
                    self.hider_pos != self.seeker_pos
                    and self.hider_pos not in self.walls
                    and self.seeker_pos not in self.walls
                ):
                    break
            return self._get_obs()

        def _get_obs(self):
            return np.array([*self.hider_pos, *self.seeker_pos])

        def step(self, hider_action, seeker_action):
            self.hider_pos = self._move_agent(hider_action, self.hider_pos)
            self.seeker_pos = self._move_agent(seeker_action, self.seeker_pos)
            reward_hider, reward_seeker, done = self._compute_reward()
            return self._get_obs(), reward_hider, reward_seeker, done

        def _move_agent(self, action, pos):
            new_pos = list(pos)
            if action == 0 and pos[0] > 0:
                new_pos[0] -= 1  # up
            elif action == 1 and pos[0] < self.grid_size - 1:
                new_pos[0] += 1  # down
            elif action == 2 and pos[1] > 0:
                new_pos[1] -= 1  # left
            elif action == 3 and pos[1] < self.grid_size - 1:
                new_pos[1] += 1  # right

            if tuple(new_pos) not in self.walls:
                return tuple(new_pos)
            return pos

        def _compute_reward(self):
            if self.hider_pos == self.seeker_pos:
                return -10, 10, True  # seeker wins
            elif self.hider_pos == self.end:
                return 10, -10, True  # hider wins by reaching the end
            else:
                return (
                    -0.1,
                    -0.1,
                    False,
                )  # small penalty for each move to encourage faster solutions

        def manhattan_distance(self, pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    class QLearningAgent:
        def __init__(self, grid_size=12, lr=0.1, gamma=0.95, epsilon=0.1):
            self.grid_size = grid_size
            self.lr = lr
            self.gamma = gamma
            self.epsilon = epsilon
            self.q_table = np.zeros((grid_size, grid_size, 4))

        def choose_action(self, state):
            if np.random.rand() < self.epsilon:
                return np.random.choice(4)
            return np.argmax(self.q_table[state[0], state[1]])

        def learn(self, state, action, reward, next_state):
            predict = self.q_table[state[2], state[3], action]
            target = reward + self.gamma * np.max(
                self.q_table[next_state[2], next_state[3]]
            )
            self.q_table[state[2], state[3], action] += self.lr * (target - predict)

        def save(self, filename):
            np.save(filename, self.q_table)

        def load(self, filename):
            self.q_table = np.load(filename)

    def train_agent(env, agent, episodes=10000):
        for episode in range(episodes):
            state = env.reset()
            done = False
            while not done:
                seeker_action = agent.choose_action(state)
                hider_action = np.random.choice(4)  # Random hider action
                next_state, reward_hider, reward_seeker, done = env.step(
                    hider_action, seeker_action
                )
                agent.learn(state, seeker_action, reward_seeker, next_state)
                state = next_state
            if episode % 1000 == 0:
                print(f"Episode {episode}/{episodes}")

    def heuristic_seeker_action(env, state):
        hider_pos = state[:2]
        seeker_pos = state[2:]
        distance = env.manhattan_distance(hider_pos, seeker_pos)
        if distance >= 4:
            if seeker_pos[0] < hider_pos[0]:
                return 1  # move down
            elif seeker_pos[0] > hider_pos[0]:
                return 0  # move up
            elif seeker_pos[1] < hider_pos[1]:
                return 3  # move right
            elif seeker_pos[1] > hider_pos[1]:
                return 2  # move left
        return np.argmax(seeker_agent.q_table[seeker_pos[0], seeker_pos[1]])

    def visualize(env, seeker_agent):
        pygame.init()
        cell_size = 50
        screen_size = env.grid_size * cell_size
        screen = pygame.display.set_mode((screen_size, screen_size))
        pygame.display.set_caption("Hide and Seek")

        font = pygame.font.Font(None, 36)

        def draw_grid():
            for x in range(0, screen_size, cell_size):
                for y in range(0, screen_size, cell_size):
                    rect = pygame.Rect(x, y, cell_size, cell_size)
                    pygame.draw.rect(screen, (200, 200, 200), rect, 1)
                for wall in env.walls:
                    rect = pygame.Rect(
                        wall[1] * cell_size, wall[0] * cell_size, cell_size, cell_size
                    )
                    pygame.draw.rect(screen, (0, 0, 0), rect)
                goal_pos = (env.end[1] * cell_size, env.end[0] * cell_size)
                pygame.draw.rect(
                    screen,
                    (0, 255, 0),
                    (goal_pos[0], goal_pos[1], cell_size, cell_size),
                )

        state = env.reset()
        done = False

        while True:
            screen.fill((255, 255, 255))
            draw_grid()

            hider_pos = (
                state[1] * cell_size + cell_size // 2,
                state[0] * cell_size + cell_size // 2,
            )
            seeker_pos = (
                state[3] * cell_size + cell_size // 2,
                state[2] * cell_size + cell_size // 2,
            )

            pygame.draw.circle(screen, (0, 0, 255), hider_pos, cell_size // 3)
            pygame.draw.circle(screen, (255, 0, 0), seeker_pos, cell_size // 3)

            hider_label = font.render("H", True, (0, 0, 0))
            seeker_label = font.render("S", True, (0, 0, 0))

            screen.blit(
                hider_label,
                (
                    hider_pos[0] - hider_label.get_width() // 2,
                    hider_pos[1] - hider_label.get_height() // 2,
                ),
            )
            screen.blit(
                seeker_label,
                (
                    seeker_pos[0] - seeker_label.get_width() // 2,
                    seeker_pos[1] - seeker_label.get_height() // 2,
                ),
            )

            pygame.display.flip()

            hider_action = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        hider_action = 0
                    elif event.key == pygame.K_DOWN:
                        hider_action = 1
                    elif event.key == pygame.K_LEFT:
                        hider_action = 2
                    elif event.key == pygame.K_RIGHT:
                        hider_action = 3

            if hider_action is not None:
                seeker_action = heuristic_seeker_action(env, state)
                next_state, reward_hider, reward_seeker, done = env.step(
                    hider_action, seeker_action
                )
                state = next_state

                if done:
                    if reward_hider > 0:
                        print("Hider wins!")
                    else:
                        print("Seeker wins!")
                    time.sleep(2)
                    state = env.reset()
                    done = False

            time.sleep(0.1)

        pygame.quit()

    if __name__ == "__main__":
        env = HideAndSeekEnv()
        seeker_agent = QLearningAgent()
        visualize(env, seeker_agent)


def game_loop():
    env = HideAndSeekEnv(grid_size=12)
    hider_agent = QLearningAgent(grid_size=12)
    seeker_agent = QLearningAgent(grid_size=12)
    hider_agent.load("hider_q_table.npy")
    seeker_agent.load("seeker_q_table.npy")
    visualize(env, hider_agent, seeker_agent)
    return_to_menu = True
    if return_to_menu:
        main_menu()


# Initialize Pygame
pygame.init()

# Set up the screen
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("RL Menu")

# Load background image
background_image = pygame.image.load("background.jpg").convert()

# Colors
black = (0, 0, 0)
transparent_white = (
    255,
    255,
    255,
    16,
)  # Extremely translucent white color with alpha value

# Font
font = pygame.font.Font(None, 36)


def main_menu():
    while True:
        screen.blit(background_image, [0, 0])

        mx, my = pygame.mouse.get_pos()

        button_start = pygame.Rect(300, 200, 200, 50)
        button_quit = pygame.Rect(300, 400, 221, 50)
        button_rl_agent = pygame.Rect(300, 300, 200, 50)

        draw_rounded_rect(screen, button_start, transparent_white, 15)
        draw_rounded_rect(screen, button_quit, transparent_white, 15)
        draw_rounded_rect(screen, button_rl_agent, transparent_white, 15)

        draw_text(
            "Start Game",
            font,
            black,
            screen,
            button_start.centerx,
            button_start.centery,
        )
        draw_text(
            "RL Agent",
            font,
            black,
            screen,
            button_rl_agent.centerx,
            button_rl_agent.centery,
        )
        draw_text("Quit", font, black, screen, button_quit.centerx, button_quit.centery)

        click = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    click = True
                    if button_start.collidepoint((mx, my)):
                        game_loop()
                    elif button_rl_agent.collidepoint((mx, my)):
                        rl_agent()
                    elif button_quit.collidepoint((mx, my)):
                        pygame.quit()
                        sys.exit()

        pygame.display.update()


main_menu()
