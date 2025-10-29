import pygame
import numpy as np
import json

class RaceEnv:
    def __init__(self, grid_size=8, render_mode=None, cell_size=40,
                 custom_map_path='custom_map.json'):

        self.grid_size = grid_size
        self.render_mode = render_mode
        self.cell_size = cell_size

        self._load_custom_map(custom_map_path)

        self.current_pos = self.start_pos
        self.max_steps = 200
        self.steps = 0

        self.screen = None
        self.clock = None
        if render_mode == 'human':
            pygame.init()
            window_size = self.grid_size * self.cell_size
            self.screen = pygame.display.set_mode((window_size, window_size))
            pygame.display.set_caption("Race Environment")
            self.clock = pygame.time.Clock()

        self.colors = {
            'background': (255, 255, 255),
            'wall': (50, 50, 50),
            'start': (0, 255, 0),
            'finish': (255, 215, 0),
            'agent': (255, 0, 0),
            'grid': (200, 200, 200)
        }

    def _load_custom_map(self, map_path):
        try:
            with open(map_path, 'r') as f:
                map_data = json.load(f)

            self.track = np.array(map_data['grid'])
            self.grid_size = len(self.track)
            self.start_pos = tuple(map_data['start_pos'])
            self.finish_pos = tuple(map_data['finish_pos'])

        except Exception as e:
            raise

    def reset(self):
        self.current_pos = self.start_pos
        self.steps = 0
        return self.current_pos

    def step(self, action):
        self.steps += 1

        moves = {
            0: (0, 0),  # stay
            1: (0, 1),  # right
            2: (0, -1),  # left
            3: (-1, 0),  # up
            4: (1, 0)  # down
        }

        move = moves.get(action, (0, 0))
        new_pos = (self.current_pos[0] + move[0], self.current_pos[1] + move[1])

        # check boundaries
        if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
            return self.current_pos, -10, True  # Out of bounds

        # check if hit wall
        if self.track[new_pos[0], new_pos[1]] == 1:
            return self.current_pos, -10, True  # Hit wall

        self.current_pos = new_pos

        if self.current_pos == self.finish_pos:
            return self.current_pos, 100, True

        if self.steps >= self.max_steps:
            return self.current_pos, -50, True

        # normal step
        return self.current_pos, -1, False

    def render(self, fps=10):
        if self.render_mode != 'human' or self.screen is None:
            return

        # handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        # clear screen
        self.screen.fill(self.colors['background'])

        # draw grid
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x = col * self.cell_size
                y = row * self.cell_size

                # draw cell
                if self.track[row, col] == 1:  # Wall
                    pygame.draw.rect(self.screen, self.colors['wall'],
                                     (x, y, self.cell_size, self.cell_size))
                elif (row, col) == self.start_pos:  # Start
                    pygame.draw.rect(self.screen, self.colors['start'],
                                     (x, y, self.cell_size, self.cell_size))
                elif (row, col) == self.finish_pos:  # Finish
                    pygame.draw.rect(self.screen, self.colors['finish'],
                                     (x, y, self.cell_size, self.cell_size))

                # draw grid lines
                pygame.draw.rect(self.screen, self.colors['grid'],
                                 (x, y, self.cell_size, self.cell_size), 1)

        # draw agent
        agent_x = self.current_pos[1] * self.cell_size + self.cell_size // 2
        agent_y = self.current_pos[0] * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, self.colors['agent'],
                           (agent_x, agent_y), self.cell_size // 3)

        pygame.display.flip()
        self.clock.tick(fps)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None