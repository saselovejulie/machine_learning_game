from pacman_game import pacman
from pacman_game import pacman_utils as pac_utils
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

pacman_game = pacman.PacMan()
pacman_game.start_game()

while True:
    result, x_t1_colored = pacman_game.next_step(pac_utils.PacManActions.NOTHING)
    # time.sleep(3)
    if result:
        pacman_game = pacman.PacMan()
        pacman_game.start_game()

    x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
    x_t1 = np.reshape(x_t1, (80, 80, 1))

