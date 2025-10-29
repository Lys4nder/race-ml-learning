import json
import threading
import random
from pathlib import Path

import pygame
import numpy as np
import torch

import button
from q_learning_agent import train_agent

pygame.init()
pygame.display.set_caption("Race Learning - Map Creator")

# constants
SCREEN_W, SCREEN_H = 640, 640
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
clock = pygame.time.Clock()

# screens
MENU, DRAW, TRAINING, DEMO = "menu", "draw", "training", "demo"
current_screen = MENU

# grid and layout
GRID_N = 8
HEADER_H = 80
FOOTER_H = 120
GRID_AREA = SCREEN_H - HEADER_H - FOOTER_H
CELL = GRID_AREA // GRID_N

# state
grid = [[0] * GRID_N for _ in range(GRID_N)]  # 0 free,1 wall,2 start,3 finish
start_pos = None
finish_pos = None
is_drawing = False

# training/demo state
policy_net = None
training_thread = None
training_complete = False
total_episodes = 10000
training_stats = {"episode": 0, "total_episodes": total_episodes, "successes": 0, "avg_reward": 0, "success_rate": 0, "epsilon": 1.0}
demo_path = []
demo_pos = None
demo_step = 0
demo_done = False
demo_active = False

# colors & fonts
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
GREEN = (0, 255, 0)
GOLD = (255, 215, 0)
RED = (255, 0, 0)
BLUE = (100, 149, 237)

title_f = pygame.font.SysFont("Arial", 32)
btn_f = pygame.font.SysFont("Arial", 24)
sm_f = pygame.font.SysFont("Arial", 18)
tiny_f = pygame.font.SysFont("Arial", 14)

def make_text_img(text, w, h, bg, fg):
    surf = pygame.Surface((w, h))
    surf.fill(bg)
    t = pygame.font.SysFont("Arial", 18).render(text, True, fg)
    surf.blit(t, t.get_rect(center=(w//2, h//2)))
    pygame.draw.rect(surf, BLACK, (0, 0, w, h), 2)
    return surf

# buttons
start_img = pygame.image.load("start.png")
start_btn = button.Button(100, 200, start_img, screen, 0.5)
btn_w, btn_h = 90, 40
start_mode_btn = button.Button(10, 10, make_text_img("Start", btn_w, btn_h, GREEN, BLACK), screen, 1.0)
finish_mode_btn = button.Button(110, 10, make_text_img("Finish", btn_w, btn_h, GOLD, BLACK), screen, 1.0)
wall_btn = button.Button(210, 10, make_text_img("Wall", btn_w, btn_h, BLUE, WHITE), screen, 1.0)
erase_btn = button.Button(310, 10, make_text_img("Erase", btn_w, btn_h, GRAY, BLACK), screen, 1.0)
clear_btn = button.Button(410, 10, make_text_img("Clear", btn_w, btn_h, (255,100,100), WHITE), screen, 1.0)
train_btn = button.Button(510, 10, make_text_img("Train!", btn_w, btn_h, (255,69,0), WHITE), screen, 1.0)

# UI helpers
def draw_grid(surface, offset_y=HEADER_H, show_agent=None):
    for r in range(GRID_N):
        for c in range(GRID_N):
            x, y = c * CELL, r * CELL + offset_y
            val = grid[r][c]
            if val == 1:
                pygame.draw.rect(surface, DARK_GRAY, (x, y, CELL, CELL))
            elif val == 2:
                pygame.draw.rect(surface, GREEN, (x, y, CELL, CELL))
                surface.blit(sm_f.render("S", True, BLACK), (x + CELL//3, y + CELL//6))
            elif val == 3:
                pygame.draw.rect(surface, GOLD, (x, y, CELL, CELL))
                surface.blit(sm_f.render("F", True, BLACK), (x + CELL//3, y + CELL//6))
            pygame.draw.rect(surface, GRAY, (x, y, CELL, CELL), 1)

    if show_agent:
        ax = show_agent[1] * CELL + CELL//2
        ay = show_agent[0] * CELL + offset_y + CELL//2
        pygame.draw.circle(surface, RED, (ax, ay), CELL//3)

def save_map():
    with open("custom_map.json", "w") as f:
        json.dump({"grid": grid, "start_pos": start_pos, "finish_pos": finish_pos}, f)

def training_progress_callback(_, stats):
    global training_stats
    training_stats = stats

def run_training_thread():
    global policy_net, training_complete
    save_map()
    policy_net, _, _, _ = train_agent(episodes=total_episodes, verbose=False, progress_callback=training_progress_callback)
    training_complete = True

def init_training():
    global training_thread, training_complete, training_stats
    training_complete = False
    training_stats = {"episode": 0, "total_episodes": total_episodes, "successes": 0, "avg_reward": 0, "success_rate": 0, "epsilon": 1.0}
    training_thread = threading.Thread(target=run_training_thread, daemon=True)
    training_thread.start()

# demo generation: rollouts to find shortest successful path, save winning_run.json
def generate_winning_path(num_attempts=20, max_steps=200):
    global demo_path, demo_step, demo_done, demo_active, demo_pos, training_stats
    if policy_net is None:
        return

    from race_env import RaceEnv
    from q_learning_agent import EnhancedRaceEnv

    policy_net_cpu = policy_net.cpu()
    policy_net_cpu.eval()

    best_path, best_len, best_reward, best_success = [], float('inf'), -float('inf'), False

    for attempt in range(num_attempts):
        base = RaceEnv(render_mode=None)
        env = EnhancedRaceEnv(base)
        state = env.reset()
        path = [[int(state[0]), int(state[1])]]
        total_reward = 0
        epsilon = 0.05 if attempt > 0 else 0.0

        for _ in range(max_steps):
            with torch.no_grad():
                s_t = torch.tensor(np.array(state, dtype=np.float32)).unsqueeze(0)
                q = policy_net_cpu(s_t)
                action = random.randint(0, 4) if random.random() < epsilon else int(torch.argmax(q[0]).item())

            state, reward, done = env.step(action)
            path.append([int(state[0]), int(state[1])])
            total_reward += reward
            if done:
                break

        reached = env.env.finish_pos is not None and env.env.current_pos == env.env.finish_pos
        L = len(path) - 1
        if reached and (not best_success or L < best_len):
            best_path, best_len, best_reward, best_success = path.copy(), L, total_reward, True
        elif not best_success and total_reward > best_reward:
            best_path, best_len, best_reward = path.copy(), L, total_reward

        if best_success and best_len <= 10 and attempt >= 5:
            break

    demo_path = best_path
    demo_step = 0
    demo_done = False
    demo_active = bool(demo_path)
    training_stats["final_reward"] = float(best_reward)
    training_stats["demo_success"] = bool(best_success)

    if best_success:
        with open("winning_run.json", "w") as f:
            json.dump(demo_path, f)
        print(f"Winning run saved ({best_len} steps)")

def init_demo():
    global demo_step, demo_done, demo_active
    generate_winning_path()
    demo_step = 0
    demo_done = False
    if demo_path:
        demo_active = True

def step_demo():
    global demo_step, demo_done
    if demo_done or not demo_path:
        return None
    if demo_step < len(demo_path) - 1:
        demo_step += 1
        return demo_path[demo_step]
    else:
        demo_done = True
        return demo_path[-1]

running = True
while running:
    dt = clock.tick(60) / 1000.0

    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False
        if current_screen == DRAW:
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if pygame.mouse.get_pos()[1] > HEADER_H:
                    is_drawing = True
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                is_drawing = False

    # draw screen
    screen.fill(WHITE)

    if current_screen == MENU:
        # gradient background
        for i in range(SCREEN_H):
            color_val = int(240 - (i / SCREEN_H) * 40)
            pygame.draw.line(screen, (color_val, color_val, 255), (0, i), (SCREEN_W, i))

        # title with shadow for depth
        title_text = "Machine Learning: Race"
        shadow_offset = 3
        shadow = title_f.render(title_text, True, (50, 50, 50))
        screen.blit(shadow, shadow.get_rect(center=(SCREEN_W // 2 + shadow_offset, 120 + shadow_offset)))

        title = title_f.render(title_text, True, WHITE)
        screen.blit(title, title.get_rect(center=(SCREEN_W // 2, 120)))

        # decorative white box with border and shadow
        box_w, box_h = 500, 220
        box_x = (SCREEN_W - box_w) // 2
        box_y = 180
        pygame.draw.rect(screen, (100, 100, 150), (box_x + 5, box_y + 5, box_w, box_h), border_radius=15)
        pygame.draw.rect(screen, WHITE, (box_x, box_y, box_w, box_h), border_radius=15)
        pygame.draw.rect(screen, BLUE, (box_x, box_y, box_w, box_h), 3, border_radius=15)

        subtitle = btn_f.render("Draw Your Custom Track", True, BLUE)
        screen.blit(subtitle, subtitle.get_rect(center=(SCREEN_W // 2, box_y + 45)))

        desc_lines = [
            "Create your own 8x8 track layout",
            "Train a DQN agent with A*-shaped rewards",
            "Then watch it find the optimal path!"
        ]
        y_offset = box_y + 85
        for line in desc_lines:
            desc = sm_f.render(line, True, (60, 60, 60))
            screen.blit(desc, desc.get_rect(center=(SCREEN_W // 2, y_offset)))
            y_offset += 28

        # centered Start button below the box
        start_btn.rect.center = (SCREEN_W // 2, box_y + box_h + 60)
        if start_btn.draw():
            current_screen = DRAW

        # footer and version text
        footer = tiny_f.render("Enhanced DQN with Deep Reinforcement Learning", True, (150, 150, 150))
        screen.blit(footer, footer.get_rect(center=(SCREEN_W // 2, SCREEN_H - 30)))

        version = tiny_f.render("v1.0 - A* Reward Shaping", True, (150, 150, 150))
        screen.blit(version, version.get_rect(center=(SCREEN_W // 2, SCREEN_H - 10)))

    elif current_screen == DRAW:
        pygame.draw.rect(screen, (240,240,240), (0, 0, SCREEN_W, HEADER_H))
        pygame.draw.line(screen, BLACK, (0, HEADER_H), (SCREEN_W, HEADER_H), 2)

        if start_mode_btn.draw():
            mode = 2
        if finish_mode_btn.draw():
            mode = 3
        if wall_btn.draw():
            mode = 1
        if erase_btn.draw():
            mode = 0
        if clear_btn.draw():
            grid = [[0]*GRID_N for _ in range(GRID_N)]
            start_pos = None
            finish_pos = None
        if train_btn.draw():
            if not start_pos or not finish_pos:
                pass
            else:
                save_map()
                init_training()
                current_screen = TRAINING

        # drawing logic
        if is_drawing:
            mx, my = pygame.mouse.get_pos()
            if my > HEADER_H:
                gx = mx // CELL
                gy = (my - HEADER_H) // CELL
                if 0 <= gx < GRID_N and 0 <= gy < GRID_N:
                    if mode == 1:
                        if grid[gy][gx] in (2,3):
                            if grid[gy][gx] == 2: start_pos = None
                            else: finish_pos = None
                        grid[gy][gx] = 1
                    elif mode == 0:
                        if grid[gy][gx] == 2: start_pos = None
                        elif grid[gy][gx] == 3: finish_pos = None
                        grid[gy][gx] = 0
                    elif mode == 2:
                        if start_pos:
                            grid[start_pos[0]][start_pos[1]] = 0
                        grid[gy][gx] = 2
                        start_pos = (gy, gx)
                    elif mode == 3:
                        if finish_pos:
                            grid[finish_pos[0]][finish_pos[1]] = 0
                        grid[gy][gx] = 3
                        finish_pos = (gy, gx)

        draw_grid(screen)

        inst = sm_f.render("ESC: Back to Menu | Click and drag to draw", True, BLACK)
        screen.blit(inst, (10, SCREEN_H - 25))
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            current_screen = MENU

    elif current_screen == TRAINING:
        # progress bar and stats
        prog = training_stats["episode"] / training_stats["total_episodes"] if training_stats["total_episodes"] else 0
        bx, by, bw, bh = (SCREEN_W - 400)//2, 200, 400, 30
        pygame.draw.rect(screen, GRAY, (bx, by, bw, bh), 2)
        pygame.draw.rect(screen, GREEN, (bx, by, int(bw * prog), bh))

        screen.blit(title_f.render("Training...", True, BLACK), (SCREEN_W//2 - 60, 100))
        screen.blit(btn_f.render(f"Episode: {training_stats['episode']}/{training_stats['total_episodes']}", True, BLACK), (SCREEN_W//2 - 120, 250))
        screen.blit(sm_f.render(f"Successes: {training_stats['successes']}", True, BLACK), (SCREEN_W//2 - 120, 300))
        screen.blit(sm_f.render(f"Last 100 Runs Success Rate: {training_stats['success_rate']:.1f}%", True, BLACK), (SCREEN_W//2 - 120, 330))
        screen.blit(sm_f.render(f"Avg Reward: {training_stats['avg_reward']:.1f}", True, BLACK), (SCREEN_W//2 - 120, 360))
        screen.blit(sm_f.render(f"Epsilon: {training_stats['epsilon']:.3f}", True, BLACK), (SCREEN_W//2 - 120, 390))

        if training_complete:
            current_screen = DEMO
            init_demo()

    elif current_screen == DEMO:
        if demo_active and not demo_done:
            demo_step_time = step_demo()
            if demo_step_time:
                demo_pos = demo_step_time

        draw_grid(screen, offset_y=HEADER_H, show_agent=demo_pos)

        # header and title
        pygame.draw.rect(screen, (240,240,240), (0, 0, SCREEN_W, HEADER_H))
        pygame.draw.line(screen, BLACK, (0, HEADER_H), (SCREEN_W, HEADER_H), 2)
        demo_success = training_stats.get("demo_success", False)
        title_text = "Winning Path" if demo_success else "Best Attempt"
        color = GREEN if demo_success else (255, 140, 0)
        screen.blit(title_f.render(title_text, True, color), (SCREEN_W//2 - 120, 25))

        # footer stats
        footer_y = HEADER_H + GRID_AREA
        pygame.draw.rect(screen, (240,240,240), (0, footer_y, SCREEN_W, FOOTER_H))
        pygame.draw.line(screen, BLACK, (0, footer_y), (SCREEN_W, footer_y), 2)
        stats_y = footer_y + 15
        step_text = f"Step: {demo_step}/{max(1, len(demo_path)-1)}"
        if demo_done:
            step_text += " - COMPLETE!" if demo_success else " - Did not finish"
        screen.blit(btn_f.render(step_text, True, GREEN if demo_success else (255,140,0)), (20, stats_y))

        screen.blit(sm_f.render(f"Training Episodes: {total_episodes}", True, BLUE), (20, stats_y + 35))
        success_pct = training_stats['success_rate'] * 100
        screen.blit(sm_f.render(f"Success Rate: {training_stats['successes']}/{total_episodes} ({success_pct:.1f}%)", True, BLUE), (320, stats_y + 35))
        screen.blit(sm_f.render(f"Final Avg Reward: {training_stats['avg_reward']:.1f}", True, BLUE), (20, stats_y + 60))
        screen.blit(sm_f.render(f"Path Reward: {training_stats.get('final_reward', 0):.1f}", True, BLUE), (320, stats_y + 60))

        inst = tiny_f.render("ESC: Back to Editor | R: Replay", True, BLACK)
        screen.blit(inst, (SCREEN_W//2 - inst.get_width()//2, SCREEN_H - 15))

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            current_screen = DRAW
            demo_active = False
        elif keys[pygame.K_r]:
            init_demo()

    pygame.display.flip()

pygame.quit()
