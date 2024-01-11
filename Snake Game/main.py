import pygame
import json
import random

# INSTRUCTIONS
print("\nINSTRUCTIONS")
print("\nW TO MOVE UP, S TO MOVE DOWN, A TO MOVE LEFT, D TO MOVE RIGHT")
print("R TO RESTART ONCE DEAD")

# INITIALIZE PYGAME
pygame.init()

# SETUP BACKGROUND 
backgroundWidth = 500
backgroundHeight = 500
background = pygame.display.set_mode((backgroundWidth, backgroundHeight))
pygame.display.set_caption("SNAKE")
# HIDE MOUSE CURSOR
pygame.mouse.set_visible(False)

# SET UP CLOCK - KEEPS TRACK OF TIME
clock = pygame.time.Clock()

# CREATE GRIDS ON BACKGROUND
def grids():
  # LOOP SO GRIDS COVER THE ENTIRE BACKGROUND
  for x in range(0, backgroundWidth, 20):
    for y in range(0, backgroundHeight, 20):
      pygame.draw.rect(background, "black", (x, y, 20, 20), 1)

# APPLE CLASS
class Apple:
  def __init__(apple, x, y, w, h):
    apple.x = x
    apple.y = y
    apple.w = w
    apple.h = h

  # DRAW APPLE METHOD
  def drawApple(apple):
    pygame.draw.rect(background, "red", (apple.x, apple.y, apple.w, apple.h))

# DRAW SNAKE
def drawSnake(arr):
  # DRAW DARK GREEN RECTANGLE AT X = POS[0] Y = POS [1] FOR EVERY ELEMENT IN SNAKE ARRAY
  for pos in arr:
    pygame.draw.rect(background, "darkgreen", (pos[0], pos[1], 20, 20))

# DRAW TEXT
def text(message, colour, x, y):
  font = pygame.font.SysFont("comicsansms", 25)  
  message = font.render(message, True, colour)
  # DRAWS TEXT ON BACKGROUND
  background.blit(message, (x, y))

# LOAD HIGHSCORE FROM TXT FILE
def loadHighScore():
  file = open("highscore.txt", "r")
  data = json.load(file)
  file.close()
  return data

# SAVE NEW HIGHSCORE TO TXT FILE
def save(anArr):
  file = open("highscore.txt", "w")
  json.dump(anArr, file)
  file.close()
  
# MAIN LOOP
def main():
  # INITIALIZE APPLEX AND APPLEY TO BE AT A RANDOM POSITION
  appleX = random.randrange(0, 480, 20)
  appleY = random.randrange(0, 480, 20)

  # INITIALIZE SNAKEHEADX AND SNAKEHEADY TO BE AT A RANDOM POSITION
  snakeHeadX = random.randrange(20, 480, 20)
  snakeHeadY = random.randrange(20, 480, 20)
  # VARIABLES TO KEEP TRACK OF SNAKEHEAD DIRECTION
  speedX = 0
  speedY = 0
  # SNAKE ARRAY - BECOMES 2D AFTER SNAKE COLLIDES WITH APPLE AND GAINS ANOTHER SEGMENT
  snakeBody = []
  # VARIABLE TO KEEP TRACK OF SNAKE LENGTH
  snakeLength = len(snakeBody) + 1

  # GAME OVER BOOLEAN VARIABLE
  gameOver = False

  # LOAD HIGH SCORE FROM TXT FILE
  highscore = loadHighScore()
  
  # WHILE TRUE (LOOP FOREVER)
  while True:
    # SET FRAMERATE FOR GAME TO 12 FPS - SLOW DOWN GAME
    clock.tick(12)
    
    # SETUP BACKGROUND
    background.fill("white")
    grids()

    # DRAW SCORE & HIGHSCORE
    score = snakeLength - 1
    text("SCORE: " + str(score), "black", 400, 0)
    
    # DRAW APPLE
    apple = Apple(appleX, appleY, 20, 20)
    apple.drawApple()

    # DRAW SNAKE - ADD SNAKE HEAD TO SNAKEBODY ARRAY (ADD NEW SEGMENTS TO BODY ARRAY)
    drawSnake(snakeBody)
    snakeBody.append([snakeHeadX, snakeHeadY])
    # IF LENGTH OF SNAKE ARRAY  > SNAKE LENGTH - DELETE FIRST ELEMENT IN SNAKE ARRAY (STOP SNAKE FROM GROWING INFINITLY)
    if len(snakeBody) > snakeLength:
      del snakeBody[0]
    # PUT SNAKE IN CONSTANT MOTION 
    snakeHeadX += speedX
    snakeHeadY += speedY
    
    # USER INPUT - CHANGE DIRECTION IF USER PRESSES A VALID KEY 
    for event in pygame.event.get():
      if event.type == pygame.KEYDOWN:
        # IF USER PRESSES W - SNAKE GOES UP
        if event.key == pygame.K_w:
          speedX = 0
          speedY = -20
        # IF USER PRESSES S - SNAKE GOES DOWN
        elif event.key == pygame.K_s:
          speedX = 0
          speedY = 20
        # IF USER PRESSES A - SNAKE GOES LEFT
        if event.key == pygame.K_a:
          speedX = -20
          speedY = 0
        # IF USER PRESSES D - SNAKE GOES RIGHT
        elif event.key == pygame.K_d:
          speedX = 20
          speedY = 0
        
    # COLLISION DETECTION - IF SNAKE HITS APPLE SNAKE GROWS BY 1 UNIT
    if snakeHeadX == apple.x and snakeHeadY == apple.y:
      snakeLength += 1
      # APPLE GOES TO ANOTHER RANDOM POSITION ON MAP
      appleX = random.randrange(0, 480, 20)
      appleY = random.randrange(0, 480, 20)
    # IF SNAKE HITS BORDER - GAME OVER 
    elif snakeHeadX >= backgroundWidth or snakeHeadY >= backgroundHeight or snakeHeadX < 0 or snakeHeadY < 0:
      gameOver = True
       
    # COLLISION WITH BODY 
    # EVERY ARRAY IN SNAKEBODY EXCEPT HEAD
    for arr in snakeBody[1:]:
      # IF ARRAY IN SNAKEBODY = SNAKEHEAD ARRAY - GAMEOVER
      if arr == [snakeHeadX, snakeHeadY]:
        gameOver = True
       
    # GAME OVER LOOP
    while gameOver == True:
      background.fill("white")
      # DISPLAY GAMEOVER, SCORE
      text("GAMEOVER", "black", 200, 200)
      text("SCORE: " + str(score), "black", 200, 220)
      # REPLACE HIGH SCORE IF SCORE IS GREATER
      if score > highscore:
        highscore = score
        save(highscore)
      # DISPLAY HIGHSCORE
      text("HIGHSCORE: " + str(highscore), "black", 200, 240)
      # UPDATE BACKGROUND
      pygame.display.update()

      # RESTART MAIN LOOP
      for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
          # IF USER PRESSES R - RESTART MAIN LOOP
          if event.key == pygame.K_r:
            main()

    # UPDATE BACKGROUND
    pygame.display.update()

# MAIN LOOP
main()