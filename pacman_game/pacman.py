import pygame
from pacman_game import pacman_utils

black = (0, 0, 0)
white = (255, 255, 255)
blue = (0, 0, 255)
green = (0, 255, 0)
red = (255, 0, 0)
purple = (255, 0, 255)
yellow = (255, 255, 0)


# This class represents the bar at the bottom that the player controls
class Wall(pygame.sprite.Sprite):
    # Constructor function
    def __init__(self, x, y, width, height, color):
        # Call the parent's constructor
        pygame.sprite.Sprite.__init__(self)

        # Make a blue wall, of the size specified in the parameters
        self.image = pygame.Surface([width, height])
        self.image.fill(color)

        # Make our top-left corner the passed-in location.
        self.rect = self.image.get_rect()
        self.rect.top = y
        self.rect.left = x


# This class represents the ball
# It derives from the "Sprite" class in Pygame
class Block(pygame.sprite.Sprite):

    # Constructor. Pass in the color of the block,
    # and its x and y position
    def __init__(self, color, width, height):
        # Call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self)

        # Create an image of the block, and fill it with a color.
        # This could also be an image loaded from the disk.
        self.image = pygame.Surface([width, height])
        self.image.fill(white)
        self.image.set_colorkey(white)
        pygame.draw.ellipse(self.image, color, [0, 0, width, height])

        # Fetch the rectangle object that has the dimensions of the image
        # image.
        # Update the position of this object by setting the values
        # of rect.x and rect.y
        self.rect = self.image.get_rect()


# This class represents the bar at the bottom that the player controls
class Player(pygame.sprite.Sprite):

    # Set speed vector
    change_x = 0
    change_y = 0

    # Constructor function
    def __init__(self, x, y, filename):
        # Call the parent's constructor
        pygame.sprite.Sprite.__init__(self)

        # Set height, width
        self.image = pygame.image.load(filename).convert()

        # Make our top-left corner the passed-in location.
        self.rect = self.image.get_rect()
        self.rect.top = y
        self.rect.left = x
        self.prev_x = x
        self.prev_y = y

    # Clear the speed of the player
    def prevdirection(self):
        self.prev_x = self.change_x
        self.prev_y = self.change_y

    # Change the speed of the player
    def changespeed(self, x, y):
        self.change_x += x
        self.change_y += y

    # Change the speed of the player
    def reset_speed(self, x, y):
        self.change_x = x
        self.change_y = y

    # Find a new position for the player
    def update(self, walls, gate):
        # Get the old position, in case we need to go back to it

        old_x = self.rect.left
        new_x = old_x+self.change_x
        prev_x = old_x+self.prev_x
        self.rect.left = new_x

        old_y = self.rect.top
        new_y = old_y+self.change_y
        prev_y = old_y+self.prev_y

        # Did this update cause us to hit a wall?
        x_collide = pygame.sprite.spritecollide(self, walls, False)
        if x_collide:
            # Whoops, hit a wall. Go back to the old position
            self.rect.left = old_x
            # self.rect.top=prev_y
            # y_collide = pygame.sprite.spritecollide(self, walls, False)
            # if y_collide:
            #     # Whoops, hit a wall. Go back to the old position
            #     self.rect.top=old_y
            #     print('a')
        else:

            self.rect.top = new_y

            # Did this update cause us to hit a wall?
            y_collide = pygame.sprite.spritecollide(self, walls, False)
            if y_collide:
                # Whoops, hit a wall. Go back to the old position
                self.rect.top = old_y
                # self.rect.left=prev_x
                # x_collide = pygame.sprite.spritecollide(self, walls, False)
                # if x_collide:
                #     # Whoops, hit a wall. Go back to the old position
                #     self.rect.left=old_x
                #     print('b')

        if gate is not False:
            gate_hit = pygame.sprite.spritecollide(self, gate, False)
            if gate_hit:
                self.rect.left = old_x
                self.rect.top = old_y


# inheritime Player klassist
class Ghost(Player):
    def __init__(self, x, y, filename, move_list, ghost):
        Player.__init__(self, x, y, filename)
        self.list = move_list
        self.ghost = ghost
        self.turn = 0
        self.steps = 0
        self.l = len(move_list) - 1

    # Change the speed of the ghost
    def change_speed(self, not_update=False):
        backup_turn = self.turn
        backup_steps = self.steps
        try:
            z = self.list[self.turn][2]
            if self.steps < z:
                self.change_x = self.list[self.turn][0]
                self.change_y = self.list[self.turn][1]
                self.steps += 1
            else:
                if self.turn < self.l:
                    self.turn += 1
                elif self.ghost == "clyde":
                    self.turn = 2
                else:
                    self.turn = 0
                self.change_x = self.list[self.turn][0]
                self.change_y = self.list[self.turn][1]
                self.steps = 0

            if not_update:
                self.turn = backup_turn
                self.steps = backup_steps

        except IndexError:
            print("IndexError!!")
            self.turn = 0
            self.steps = 0


# This creates all the walls in room 1
def setup_room_one(all_sprites_list):
    # Make the walls. (x_pos, y_pos, width, height)
    wall_list = pygame.sprite.RenderPlain()

    # This is a list of walls. Each is in the form [x, y, width, height]
    walls = pacman_utils.walls

    # Loop through the list. Create the wall, add it to the list
    for item in walls:
        wall = Wall(item[0], item[1], item[2], item[3], blue)
        wall_list.add(wall)
        all_sprites_list.add(wall)

    # return our new list
    return wall_list


def setup_gate(all_sprites_list):
    gate = pygame.sprite.RenderPlain()
    gate.add(Wall(282, 242, 42, 2, white))
    all_sprites_list.add(gate)
    return gate


class PacMan:
    def __init__(self):
        self.trollIcon = pygame.image.load('images/Trollman.png')

        # Call this function so the Pygame library can initialize itself
        pygame.init()

        # Create an 606x606 sized screen
        self.screen = pygame.display.set_mode([606, 606])

        # This is a list of 'sprites.' Each block in the program is
        # added to this list. The list is managed by a class called 'RenderPlain.'

        # Set the title of the window
        pygame.display.set_caption('Pacman')

        # Create a surface we can draw on
        background = pygame.Surface(self.screen.get_size())

        # Used for converting color maps and such
        background = background.convert()

        # Fill the screen with a black background
        background.fill(black)

        self.clock = pygame.time.Clock()

        pygame.font.init()
        self.font = pygame.font.Font("freesansbold.ttf", 24)

        # default locations for Pacman and monsters
        self.w = 303 - 16  # Width
        self.p_h = (7 * 60) + 19  # Pacman height
        self.m_h = (4 * 60) + 19  # Monster height
        self.b_h = (3 * 60) + 19  # Binky height
        self.i_w = 303 - 16 - 32  # Inky width
        self.c_w = 303 + (32 - 16)  # Clyde width

        self.all_sprites_list = pygame.sprite.RenderPlain()
        self.block_list = pygame.sprite.RenderPlain()
        self.monster_list = pygame.sprite.RenderPlain()
        self.pacman_collide = pygame.sprite.RenderPlain()

        self.wall_list = setup_room_one(self.all_sprites_list)

        self.gate = setup_gate(self.all_sprites_list)

        # Create the player paddle object
        self.pacman = Player(self.w, self.p_h, "images/Trollman.png")
        self.all_sprites_list.add(self.pacman)
        self.pacman_collide.add(self.pacman)

        self.blinky = Ghost(self.w, self.b_h, "images/Blinky.png", pacman_utils.blinky_directions, False)
        self.monster_list.add(self.blinky)
        self.all_sprites_list.add(self.blinky)

        self.pinky = Ghost(self.w, self.m_h, "images/Pinky.png", pacman_utils.pinky_directions, False)
        self.monster_list.add(self.pinky)
        self.all_sprites_list.add(self.pinky)

        self.inky = Ghost(self.i_w, self.m_h, "images/Inky.png", pacman_utils.inky_directions, False)
        self.monster_list.add(self.inky)
        self.all_sprites_list.add(self.inky)

        self.clyde = Ghost(self.c_w, self.m_h, "images/Clyde.png", pacman_utils.clyde_directions, "clyde")
        self.monster_list.add(self.clyde)
        self.all_sprites_list.add(self.clyde)

        self.bll = 0
        self.score = 0

    def start_game(self):

        pygame.display.set_icon(self.trollIcon)
        pygame.mixer.init()
        pygame.mixer.music.load('pacman.mp3')
        pygame.mixer.music.play(-1, 0.0)

        # Draw the grid
        for row in range(19):
            for column in range(19):
                if (row == 7 or row == 8) and (column == 8 or column == 9 or column == 10):
                    continue
                else:
                    block = Block(yellow, 4, 4)

                    # Set a random location for the block
                    block.rect.x = (30*column+6)+26
                    block.rect.y = (30*row+6)+26

                    b_collide = pygame.sprite.spritecollide(block, self.wall_list, False)
                    p_collide = pygame.sprite.spritecollide(block, self.pacman_collide, False)
                    if b_collide:
                        continue
                    elif p_collide:
                        continue
                    else:
                        # Add the block to the list of objects
                        self.block_list.add(block)
                        self.all_sprites_list.add(block)

        self.bll = len(self.block_list)

    def next_step(self, action):

        # 奖励机制
        reward = 0.1

        # 游戏结束标识
        game_over = False

        # ALL EVENT PROCESSING SHOULD GO BELOW THIS COMMENT
        if action == pacman_utils.PacManActions.LEFT:
            self.pacman.reset_speed(-30, 0)
        if action == pacman_utils.PacManActions.RIGHT:
            self.pacman.reset_speed(30, 0)
        if action == pacman_utils.PacManActions.UP:
            self.pacman.reset_speed(0, -30)
        if action == pacman_utils.PacManActions.DOWN:
            self.pacman.reset_speed(0, 30)
        if action == pacman_utils.PacManActions.NOTHING:
            self.pacman.reset_speed(0, 0)

        # ALL EVENT PROCESSING SHOULD GO ABOVE THIS COMMENT

        # ALL GAME LOGIC SHOULD GO BELOW THIS COMMENT
        self.pacman.update(self.wall_list, self.gate)

        self.pinky.change_speed()
        self.pinky.change_speed(True)
        self.pinky.update(self.wall_list, False)

        self.blinky.change_speed()
        self.blinky.change_speed(True)
        self.blinky.update(self.wall_list, False)

        self.inky.change_speed()
        self.inky.change_speed(True)
        self.inky.update(self.wall_list, False)

        self.clyde.change_speed()
        self.clyde.change_speed(True)
        self.clyde.update(self.wall_list, False)

        # See if the Pacman block has collided with anything.
        blocks_hit_list = pygame.sprite.spritecollide(self.pacman, self.block_list, True)

        # Check the list of collisions.
        if len(blocks_hit_list) > 0:
            get_score = len(blocks_hit_list)
            self.score += get_score
            # 吃到豆子reward增加
            if get_score > 0:
                reward = 1

        # ALL GAME LOGIC SHOULD GO ABOVE THIS COMMENT
        # ALL CODE TO DRAW SHOULD GO BELOW THIS COMMENT
        self.screen.fill(black)

        self.wall_list.draw(self.screen)
        self.gate.draw(self.screen)
        self.all_sprites_list.draw(self.screen)
        self.monster_list.draw(self.screen)

        text = self.font.render("Score: "+str(self.score)+"/"+str(self.bll), True, red)
        self.screen.blit(text, [10, 10])

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())

        if self.score == self.bll:
            game_over = True

        monster_hit_list = pygame.sprite.spritecollide(self.pacman, self.monster_list, False)

        if monster_hit_list:
            # 碰到怪物, reward降低, 游戏结束
            reward = -1
            game_over = True

        # ALL CODE TO DRAW SHOULD GO ABOVE THIS COMMENT
        pygame.display.flip()

        self.clock.tick(10)

        return game_over, image_data, reward
