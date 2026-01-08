import pygame
pygame.init()



WIDTH = 800
HEIGHT = 400
PANEL_HEIGHT = 100

screen = pygame.display.set_mode((WIDTH, HEIGHT + PANEL_HEIGHT))

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY  = (100, 100, 100)
BLUE  = (50, 150, 255)
RED   = (255, 50, 50)

screen.fill(BLACK)

big_font = pygame.font.Font(None, 48)
small_font = pygame.font.Font(None, 24)

PADDLE_WIDTH = 12
PADDLE_HEIGHT = 80

left_paddle = pygame.Rect(30, (HEIGHT - PADDLE_HEIGHT) // 2,
                           PADDLE_WIDTH, PADDLE_HEIGHT)

right_paddle = pygame.Rect(WIDTH - 30 - PADDLE_WIDTH,(HEIGHT - PADDLE_HEIGHT) // 2,
                            PADDLE_WIDTH, PADDLE_HEIGHT)

score_left = 0
score_right = 0
rally = 0

clock = pygame.time.Clock()
running = True

#esc
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False


    # middle line
    for y in range(0, HEIGHT, 20):
        pygame.draw.rect(screen, GRAY, (WIDTH // 2 - 2, y, 4, 10))

    pygame.draw.rect(screen, BLUE, left_paddle)
    pygame.draw.rect(screen, RED, right_paddle)
    rally_text = small_font.render(f"Rally: {rally}", True, WHITE)
    screen.blit(rally_text, (WIDTH // 2 - 40, HEIGHT - 30))

    # bottom
    pygame.draw.rect(screen, (20, 20, 20),
                     (0, HEIGHT, WIDTH, PANEL_HEIGHT))
    pygame.draw.line(screen, WHITE,
                     (0, HEIGHT), (WIDTH, HEIGHT), 2)


    pygame.display.flip()
    clock.tick(60)

pygame.quit()
