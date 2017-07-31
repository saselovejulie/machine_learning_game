import cv2
import numpy as np

from pacman_game import pacman_utils as pac_utils
from pacman_game import pacman
import time
import matplotlib.pyplot as plt

# pacman_game = pacman.PacMan()
# pacman_game.start_game()
#
# while True:
#     result, x_t1_colored, reward = pacman_game.next_step(pac_utils.PacManActions.LEFT)
#     print(result, reward)
#     # time.sleep(1)
#
#     # 修改图片色值域
#     x_t = cv2.cvtColor(cv2.resize(x_t1_colored, (101, 101)), cv2.COLOR_BGR2HLS)
#     # 像素高于阈值时，给像素赋予新值，否则，赋予另外一种颜色
#     ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
#     s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
#     print(s_t[:, :, :3])
#     print(s_t)
#     plt.imshow(x_t1)
#     plt.show()
#     x_t1 = np.reshape(x_t1, (101, 101, 3))


a = np.zeros((101, 101, 3))
print(a.shape)
print(np.append(a, a[:, :, :1], axis=2).shape)
print(np.stack((a, a, a, a), axis=2).shape)
# print(a[:, :, :3, :])
