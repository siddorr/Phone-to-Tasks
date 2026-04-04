#!/usr/bin/env python3
import pygame
import sys

# --- Config ---
WIDTH, HEIGHT = 800, 400
PADDLE_W, PADDLE_H = 10, 60
BALL_SIZE = 12
PADDLE_SPEED = 5
BALL_SPEED_X = 4
BALL_SPEED_Y = 4
FPS = 60

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    # Paddles
    left_y = HEIGHT // 2 - PADDLE_H // 2
    right_y = HEIGHT // 2 - PADDLE_H // 2

    # Ball
    bx = WIDTH // 2
    by = HEIGHT // 2
    vx = BALL_SPEED_X
    vy = BALL_SPEED_Y

    score_left = 0
    score_right = 0

    font = pygame.font.SysFont(None, 32)

    while True:
        # --- Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        keys = pygame.key.get_pressed()

        # Left paddle: W / S
        if keys[pygame.K_w]:
            left_y -= PADDLE_SPEED
        if keys[pygame.K_s]:
            left_y += PADDLE_SPEED

        # Right paddle: UP / DOWN arrows
        if keys[pygame.K_UP]:
            right_y -= PADDLE_SPEED
        if keys[pygame.K_DOWN]:
            right_y += PADDLE_SPEED

        # Clamp paddle positions
        left_y = max(0, min(HEIGHT - PADDLE_H, left_y))
        right_y = max(0, min(HEIGHT - PADDLE_H, right_y))

        # --- Ball movement ---
        bx += vx
        by += vy

        # --- Paddle rectangles and fixed collision ---
        left_rect  = pygame.Rect(10, left_y, PADDLE_W, PADDLE_H)
        right_rect = pygame.Rect(WIDTH - 20, right_y, PADDLE_W, PADDLE_H)
        ball_rect  = pygame.Rect(bx, by, BALL_SIZE, BALL_SIZE)

        # Collision left
        if ball_rect.colliderect(left_rect) and vx < 0:
            vx *= -1
            bx = left_rect.right

        # Collision right
        if ball_rect.colliderect(right_rect) and vx > 0:
            vx *= -1
            bx = right_rect.left - BALL_SIZE


        # Bounce top/bottom
        if by <= 0 or by >= HEIGHT - BALL_SIZE:
            vy *= -1

        # Bounce left paddle
        if bx <= 20 and left_y < by < left_y + PADDLE_H:
            vx *= -1

        # Bounce right paddle
        if bx >= WIDTH - 20 - BALL_SIZE and right_y < by < right_y + PADDLE_H:
            vx *= -1

        # Scoring
        if bx < 0:
            score_right += 1
            bx, by = WIDTH // 2, HEIGHT // 2
        if bx > WIDTH:
            score_left += 1
            bx, by = WIDTH // 2, HEIGHT // 2

        # --- Draw ---
        screen.fill((0, 0, 0))

        # Paddles
        pygame.draw.rect(screen, (255, 255, 255), (10, left_y, PADDLE_W, PADDLE_H))
        pygame.draw.rect(screen, (255, 255, 255), (WIDTH-20, right_y, PADDLE_W, PADDLE_H))

        # Ball
        pygame.draw.rect(screen, (255, 255, 255), (bx, by, BALL_SIZE, BALL_SIZE))

        # Score
        score_text = font.render(f"{score_left}   :   {score_right}", True, (255,255,255))
        screen.blit(score_text, (WIDTH // 2 - 40, 20))

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
