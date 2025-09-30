import numpy as np
import random
import matplotlib.pyplot as plt

plt.style.use('dark_background')

# maze settings
MAZE_SIZE = 12

# Tile types
EMPTY = 0

HORIZONTAL = 1
VERTICAL = 2

CORNER_L = 3
CORNER_REVERSE_L = 4
CORNER_7 = 5
CORNER_REVERSE_7 = 6

JUNCTION_T = 7
JUNCTION_T_90 = 8
JUNCTION_T_180 = 9
JUNCTION_T_270 = 10

JUNCTION_CROSS = 11

DEAD_END_UP = 12
DEAD_END_DOWN = 13
DEAD_END_LEFT = 14
DEAD_END_RIGHT = 15

UP = (0, 1)
DOWN = (0, -1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# plotting settings
TILE_SIZE = 9 #should be odd
TILE_SIZE_HALF = TILE_SIZE // 2
TILE_SIZE_NINTH = TILE_SIZE // 9

def add_pos(pos1, pos2):
    return (pos1[0] + pos2[0], pos1[1] + pos2[1])

def within_bounds(pos, shape):
    if pos[0] < 0 or pos[0] >= shape[0] or pos[1] < 0 or pos[1] >= shape[1]:
        return False
    return True

def check_up(piece):
    if piece == HORIZONTAL:
        return False
    elif piece == VERTICAL:
        return True
    elif piece == CORNER_L:
        return False
    elif piece == CORNER_REVERSE_L:
        return False
    elif piece == CORNER_7:
        return True
    elif piece == CORNER_REVERSE_7:
        return True
    elif piece == JUNCTION_T:
        return True
    elif piece == JUNCTION_T_90:
        return True
    elif piece == JUNCTION_T_180:
        return False
    elif piece == JUNCTION_T_270:
        return True
    elif piece == JUNCTION_CROSS:
        return True
    elif piece == DEAD_END_UP:
        return False
    elif piece == DEAD_END_DOWN:
        return True
    elif piece == DEAD_END_LEFT:
        return False
    elif piece == DEAD_END_RIGHT:
        return False

def check_down(piece):
    if piece == HORIZONTAL:
        return False
    elif piece == VERTICAL:
        return True
    elif piece == CORNER_L:
        return True
    elif piece == CORNER_REVERSE_L:
        return True
    elif piece == CORNER_7:
        return False
    elif piece == CORNER_REVERSE_7:
        return False
    elif piece == JUNCTION_T:
        return False
    elif piece == JUNCTION_T_90:
        return True
    elif piece == JUNCTION_T_180:
        return True
    elif piece == JUNCTION_T_270:
        return True
    elif piece == JUNCTION_CROSS:
        return True
    elif piece == DEAD_END_UP:
        return True
    elif piece == DEAD_END_DOWN:
        return False
    elif piece == DEAD_END_LEFT:
        return False
    elif piece == DEAD_END_RIGHT:
        return False

def check_left(piece):
    if piece == HORIZONTAL:
        return True
    elif piece == VERTICAL:
        return False
    elif piece == CORNER_L:
        return True
    elif piece == CORNER_REVERSE_L:
        return False
    elif piece == CORNER_7:
        return False
    elif piece == CORNER_REVERSE_7:
        return True
    elif piece == JUNCTION_T:
        return True
    elif piece == JUNCTION_T_90:
        return False
    elif piece == JUNCTION_T_180:
        return True
    elif piece == JUNCTION_T_270:
        return True
    elif piece == JUNCTION_CROSS:
        return True
    elif piece == DEAD_END_UP:
        return False
    elif piece == DEAD_END_DOWN:
        return False
    elif piece == DEAD_END_LEFT:
        return False
    elif piece == DEAD_END_RIGHT:
        return True

def check_right(piece):
    if piece == HORIZONTAL:
        return True
    elif piece == VERTICAL:
        return False
    elif piece == CORNER_L:
        return False
    elif piece == CORNER_REVERSE_L:
        return True
    elif piece == CORNER_7:
        return True
    elif piece == CORNER_REVERSE_7:
        return False
    elif piece == JUNCTION_T:
        return True
    elif piece == JUNCTION_T_90:
        return True
    elif piece == JUNCTION_T_180:
        return True
    elif piece == JUNCTION_T_270:
        return False
    elif piece == JUNCTION_CROSS:
        return True
    elif piece == DEAD_END_UP:
        return False
    elif piece == DEAD_END_DOWN:
        return False
    elif piece == DEAD_END_LEFT:
        return True
    elif piece == DEAD_END_RIGHT:
        return False

def get_tile(dirs):
    if len(dirs) == 1:
        if UP in dirs:
            return DEAD_END_UP
        elif DOWN in dirs:
            return DEAD_END_DOWN
        elif LEFT in dirs:
            return DEAD_END_LEFT
        elif RIGHT in dirs:
            return DEAD_END_RIGHT
    elif len(dirs) == 2:
        if UP in dirs:
            if DOWN in dirs:
                return VERTICAL
            elif LEFT in dirs:
                return CORNER_REVERSE_L
            elif RIGHT in dirs:
                return CORNER_L
        elif DOWN in dirs:
            if LEFT in dirs:
                return CORNER_7
            elif RIGHT in dirs:
                return CORNER_REVERSE_7
        elif LEFT in dirs:
            if RIGHT in dirs:
                return HORIZONTAL
    elif len(dirs) == 3:
        if UP in dirs:
            if DOWN in dirs:
                if LEFT in dirs:
                    return JUNCTION_T_90
                elif RIGHT in dirs:
                    return JUNCTION_T_270
            elif LEFT in dirs:
                if RIGHT in dirs:
                    return JUNCTION_T_180
        elif DOWN in dirs:
            if LEFT in dirs:
                if RIGHT in dirs:
                    return JUNCTION_T
    elif len(dirs) == 4:
        return JUNCTION_CROSS

    raise ValueError("Invalid direction combination")

def tile_str(piece):
    if piece == HORIZONTAL:
        return 'HORIZONTAL'
    elif piece == VERTICAL:
        return 'VERTICAL'
    elif piece == CORNER_L:
        return 'CORNER_L'
    elif piece == CORNER_REVERSE_L:
        return 'CORNER_REVERSE_L'
    elif piece == CORNER_7:
        return 'CORNER_7'
    elif piece == CORNER_REVERSE_7:
        return 'CORNER_REVERSE_7'
    elif piece == JUNCTION_T:
        return 'JUNCTION_T'
    elif piece == JUNCTION_T_90:
        return 'JUNCTION_T_90'
    elif piece == JUNCTION_T_180:
        return 'JUNCTION_T_180'
    elif piece == JUNCTION_T_270:
        return 'JUNCTION_T_270'
    elif piece == JUNCTION_CROSS:
        return 'JUNCTION_CROSS'
    elif piece == DEAD_END_UP:
        return 'DEAD_END_UP'
    elif piece == DEAD_END_DOWN:
        return 'DEAD_END_DOWN'
    elif piece == DEAD_END_LEFT:
        return 'DEAD_END_LEFT'
    elif piece == DEAD_END_RIGHT:
        return 'DEAD_END_RIGHT'
    elif piece == EMPTY:
        return 'EMPTY'

def dir_str(dir):
    if dir == UP:
        return 'up'
    elif dir == DOWN:
        return 'down'
    elif dir == LEFT:
        return 'left'
    elif dir == RIGHT:
        return 'right'

def dirs_str(dirs):
    return [dir_str(d) for d in dirs]

def dir_tuple(dir):
    if dir == 'up':
        return UP
    elif dir == 'down':
        return DOWN
    elif dir == 'left':
        return LEFT
    elif dir == 'right':
        return RIGHT

def dirs_tuple(dirs):
    return [dir_tuple(d) for d in dirs]

def get_maze_junction_ratio(maze):
    n_straight = 0
    n_t_junctions = 0
    n_dead_ends = 0
    for x in range(maze.shape[0]):
        for y in range(maze.shape[1]):
            if maze[x, y] == 0:
                pass
            if maze[x, y] == HORIZONTAL:
                n_straight += 1
            if maze[x, y] == VERTICAL:
                n_straight += 1
            if maze[x, y] == CORNER_7:
                n_straight += 1
            if maze[x, y] == CORNER_L:
                n_straight += 1
            if maze[x, y] == CORNER_REVERSE_7:
                n_straight += 1
            if maze[x, y] == CORNER_REVERSE_L:
                n_straight += 1
            if maze[x, y] == DEAD_END_UP:
                n_dead_ends += 1
            if maze[x, y] == DEAD_END_DOWN:
                n_dead_ends += 1
            if maze[x, y] == DEAD_END_LEFT:
                n_dead_ends += 1
            if maze[x, y] == DEAD_END_RIGHT:
                n_dead_ends += 1
            if maze[x, y] == JUNCTION_T:
                n_t_junctions += 1
            if maze[x, y] == JUNCTION_T_90:
                n_t_junctions += 1
            if maze[x, y] == JUNCTION_T_180:
                n_t_junctions += 1
            if maze[x, y] == JUNCTION_T_270:
                n_t_junctions += 1
            if maze[x, y] == JUNCTION_CROSS:
                n_t_junctions += 1

    if n_straight == 0:
        return 1
    
    return n_t_junctions/(n_straight+n_t_junctions+n_dead_ends)

def create_maze(desired_junction_ratio, debug=False):
    size = MAZE_SIZE

    maze = np.zeros((size, size))
    start = (0, int(np.random.randint(0, size)))
    start_dir = None

    if debug:
        print(f'start: {start}')

    possible_dirs = [UP, DOWN, LEFT, RIGHT]
    to_fill = set()
    to_fill.add(start)

    while to_fill:
        cur_pos = to_fill.pop()

        # sets containing the possible directions
        # the | set operation is used frequently
        empty = set()
        connected = set()

        if debug:
            print(f'cur pos: {cur_pos}')

        up_pos = add_pos(cur_pos, UP)
        down_pos = add_pos(cur_pos, DOWN)
        left_pos = add_pos(cur_pos, LEFT)
        right_pos = add_pos(cur_pos, RIGHT)

        for d in possible_dirs:
            if d == UP:
                if not within_bounds(up_pos, maze.shape):
                    continue
                if debug:
                    print(f'up pos: {up_pos} {tile_str(maze[up_pos[0], up_pos[1]])}')
                if maze[up_pos[0], up_pos[1]] != EMPTY:
                    if check_up(maze[up_pos[0], up_pos[1]]):
                        connected.add(UP)
                        if debug:
                            print(f'up is connected')
                else:
                    empty.add(UP)
            elif d == DOWN:
                if not within_bounds(down_pos, maze.shape):
                    continue
                if debug:
                    print(f'down pos: {down_pos} {tile_str(maze[down_pos[0], down_pos[1]])}')
                if maze[down_pos[0], down_pos[1]] != EMPTY:
                    if check_down(maze[down_pos[0], down_pos[1]]):
                        connected.add(DOWN)
                        if debug:
                            print(f'down is connected')
                else:
                    empty.add(DOWN)
            elif d == LEFT:
                if not within_bounds(left_pos, maze.shape):
                    continue
                if debug:
                    print(f'left pos: {left_pos} {tile_str(maze[left_pos[0], left_pos[1]])}')
                if maze[left_pos[0], left_pos[1]] != EMPTY:
                    if check_left(maze[left_pos[0], left_pos[1]]):
                        connected.add(LEFT)
                        if debug:
                            print(f'left is connected')
                else:
                    empty.add(LEFT)
            elif d == RIGHT:
                if not within_bounds(right_pos, maze.shape):
                    continue
                if debug:
                    print(f'right pos: {right_pos} {tile_str(maze[right_pos[0], right_pos[1]])}')
                if maze[right_pos[0], right_pos[1]] != EMPTY:
                    if check_right(maze[right_pos[0], right_pos[1]]):
                        connected.add(RIGHT)
                        if debug:
                            print(f'right is connected')
                else:
                    empty.add(RIGHT)

        if debug:
            print(f'can connect: {dirs_str(empty | connected)}')

        # check if this should be a 2-way 
        # (are there enough empty, is it a junction, are there too many junctions, will this fill the last to_fill spot with a 2-way)
        if len(empty) > 0 and len(empty | connected) >= 3 and get_maze_junction_ratio(maze) > desired_junction_ratio and len(to_fill) > 0:
            if debug:
                print(f'2 way')

            while len(empty | connected) > 2 and len(empty) > 0:
                empty.pop()

        if debug:
            print(f'will connect: {dirs_str(empty | connected)}')

        maze[cur_pos[0], cur_pos[1]] = get_tile(empty | connected)

        if cur_pos == start:
            start_dir = empty.pop()
            empty.add(start_dir)

        if debug:
            print(f'tile: {tile_str(maze[cur_pos[0], cur_pos[1]])}')

        for d in empty:
            to_fill.add(add_pos(cur_pos, d))

        if debug:
            print(f'len to_fill: {len(to_fill)}\n')

        if debug:
            plot_maze(maze, cur_pos, debug=debug)

    return maze, start, start_dir

def make_vertical_wall():
    img = np.zeros((TILE_SIZE, TILE_SIZE))
    img[:, TILE_SIZE_HALF] = 1
    return img

def make_horizontal_wall():
    img = np.zeros((TILE_SIZE, TILE_SIZE))
    img[TILE_SIZE_HALF, :] = 1
    return img

def make_corner_l():
    img = np.zeros((TILE_SIZE, TILE_SIZE))
    img[TILE_SIZE_HALF, TILE_SIZE_HALF+1:TILE_SIZE] = 1
    img[TILE_SIZE_HALF:TILE_SIZE, TILE_SIZE_HALF] = 1
    return img

def make_corner_reverse_l():
    img = np.zeros((TILE_SIZE, TILE_SIZE))
    img[TILE_SIZE_HALF, 0:TILE_SIZE_HALF + 1] = 1
    img[TILE_SIZE_HALF:TILE_SIZE, TILE_SIZE_HALF] = 1
    return img

def make_corner_7():
    img = np.zeros((TILE_SIZE, TILE_SIZE))
    img[TILE_SIZE_HALF, 0:TILE_SIZE_HALF + 1] = 1
    img[0:TILE_SIZE_HALF + 1, TILE_SIZE_HALF] = 1
    return img

def make_corner_reverse_7():
    img = np.zeros((TILE_SIZE, TILE_SIZE))
    img[TILE_SIZE_HALF, TILE_SIZE_HALF:TILE_SIZE] = 1
    img[0:TILE_SIZE_HALF + 1, TILE_SIZE_HALF] = 1
    return img

def make_junction_t():
    img = np.zeros((TILE_SIZE, TILE_SIZE))
    img[TILE_SIZE_HALF, :] = 1
    img[0:TILE_SIZE_HALF + 1, TILE_SIZE_HALF] = 1
    return img

def make_junction_t_90():
    img = np.zeros((TILE_SIZE, TILE_SIZE))
    img[:, TILE_SIZE_HALF] = 1
    img[TILE_SIZE_HALF, 0:TILE_SIZE_HALF + 1] = 1
    return img

def make_junction_t_180():
    img = np.zeros((TILE_SIZE, TILE_SIZE))
    img[TILE_SIZE_HALF, :] = 1
    img[TILE_SIZE_HALF:TILE_SIZE, TILE_SIZE_HALF] = 1
    return img

def make_junction_t_270():
    img = np.zeros((TILE_SIZE, TILE_SIZE))
    img[:, TILE_SIZE_HALF] = 1
    img[TILE_SIZE_HALF, TILE_SIZE_HALF + 1:TILE_SIZE] = 1
    return img

def make_junction_cross():
    img = np.zeros((TILE_SIZE, TILE_SIZE))
    img[:, TILE_SIZE_HALF] = 1
    img[TILE_SIZE_HALF, :] = 1
    return img

def make_dead_end_up():
    img = np.zeros((TILE_SIZE, TILE_SIZE))
    img[TILE_SIZE_HALF:TILE_SIZE, TILE_SIZE_HALF] = 1
    img[TILE_SIZE_HALF-TILE_SIZE_NINTH:TILE_SIZE_HALF+1+TILE_SIZE_NINTH, TILE_SIZE_HALF-TILE_SIZE_NINTH:TILE_SIZE_HALF+1+TILE_SIZE_NINTH] = 1
    return img

def make_dead_end_down():
    img = np.zeros((TILE_SIZE, TILE_SIZE))
    img[0:TILE_SIZE_HALF+1, TILE_SIZE_HALF] = 1
    img[TILE_SIZE_HALF-TILE_SIZE_NINTH:TILE_SIZE_HALF+1+TILE_SIZE_NINTH, TILE_SIZE_HALF-TILE_SIZE_NINTH:TILE_SIZE_HALF+1+TILE_SIZE_NINTH] = 1
    return img

def make_dead_end_left():
    img = np.zeros((TILE_SIZE, TILE_SIZE))
    img[TILE_SIZE_HALF, 0:TILE_SIZE_HALF + 1] = 1
    img[TILE_SIZE_HALF-TILE_SIZE_NINTH:TILE_SIZE_HALF+1+TILE_SIZE_NINTH, TILE_SIZE_HALF-TILE_SIZE_NINTH:TILE_SIZE_HALF+1+TILE_SIZE_NINTH] = 1
    return img

def make_dead_end_right():
    img = np.zeros((TILE_SIZE, TILE_SIZE))
    img[TILE_SIZE_HALF, TILE_SIZE_HALF:TILE_SIZE] = 1
    img[TILE_SIZE_HALF-TILE_SIZE_NINTH:TILE_SIZE_HALF+1+TILE_SIZE_NINTH, TILE_SIZE_HALF-TILE_SIZE_NINTH:TILE_SIZE_HALF+1+TILE_SIZE_NINTH] = 1
    return img

def plot_maze(maze, pos = None, debug=False):
    size = len(maze)
    fig = plt.figure(figsize=(2, 2), dpi=300)
    maze_image = np.zeros((size * TILE_SIZE, size * TILE_SIZE))
    for x in range(maze.shape[0]):
        for y in range(maze.shape[1]):
            if debug:
                plt.text(x * TILE_SIZE, y * TILE_SIZE, f'{x, y}', ha='left', va='bottom', fontsize=3)

            if pos:
                if (x, y) == pos:
                    plt.text(x * TILE_SIZE + TILE_SIZE_HALF, y * TILE_SIZE + TILE_SIZE_HALF, 'X', ha='center', va='center', fontsize=6, color='red')

            if maze[x, y] == CORNER_L:
                maze_image[y * TILE_SIZE:(y + 1) * TILE_SIZE, x * TILE_SIZE:(x + 1) * TILE_SIZE] = make_corner_l()
            elif maze[x, y] == CORNER_REVERSE_L:
                maze_image[y * TILE_SIZE:(y + 1) * TILE_SIZE, x * TILE_SIZE:(x + 1) * TILE_SIZE] = make_corner_reverse_l()
            elif maze[x, y] == CORNER_7:
                maze_image[y * TILE_SIZE:(y + 1) * TILE_SIZE, x * TILE_SIZE:(x + 1) * TILE_SIZE] = make_corner_7()
            elif maze[x, y] == CORNER_REVERSE_7:
                maze_image[y * TILE_SIZE:(y + 1) * TILE_SIZE, x * TILE_SIZE:(x + 1) * TILE_SIZE] = make_corner_reverse_7()
            elif maze[x, y] == JUNCTION_T:
                maze_image[y * TILE_SIZE:(y + 1) * TILE_SIZE, x * TILE_SIZE:(x + 1) * TILE_SIZE] = make_junction_t()
            elif maze[x, y] == JUNCTION_T_90:
                maze_image[y * TILE_SIZE:(y + 1) * TILE_SIZE, x * TILE_SIZE:(x + 1) * TILE_SIZE] = make_junction_t_90()
            elif maze[x, y] == JUNCTION_T_180:
                maze_image[y * TILE_SIZE:(y + 1) * TILE_SIZE, x * TILE_SIZE:(x + 1) * TILE_SIZE] = make_junction_t_180()
            elif maze[x, y] == JUNCTION_T_270:
                maze_image[y * TILE_SIZE:(y + 1) * TILE_SIZE, x * TILE_SIZE:(x + 1) * TILE_SIZE] = make_junction_t_270()
            elif maze[x, y] == JUNCTION_CROSS:
                maze_image[y * TILE_SIZE:(y + 1) * TILE_SIZE, x * TILE_SIZE:(x + 1) * TILE_SIZE] = make_junction_cross()
            elif maze[x, y] == HORIZONTAL:
                maze_image[y * TILE_SIZE:(y + 1) * TILE_SIZE, x * TILE_SIZE:(x + 1) * TILE_SIZE] = make_horizontal_wall()
            elif maze[x, y] == VERTICAL:
                maze_image[y * TILE_SIZE:(y + 1) * TILE_SIZE, x * TILE_SIZE:(x + 1) * TILE_SIZE] = make_vertical_wall()
            elif maze[x, y] == DEAD_END_UP:
                maze_image[y * TILE_SIZE:(y + 1) * TILE_SIZE, x * TILE_SIZE:(x + 1) * TILE_SIZE] = make_dead_end_up()
            elif maze[x, y] == DEAD_END_DOWN:
                maze_image[y * TILE_SIZE:(y + 1) * TILE_SIZE, x * TILE_SIZE:(x + 1) * TILE_SIZE] = make_dead_end_down()
            elif maze[x, y] == DEAD_END_LEFT:
                maze_image[y * TILE_SIZE:(y + 1) * TILE_SIZE, x * TILE_SIZE:(x + 1) * TILE_SIZE] = make_dead_end_left()
            elif maze[x, y] == DEAD_END_RIGHT:
                maze_image[y * TILE_SIZE:(y + 1) * TILE_SIZE, x * TILE_SIZE:(x + 1) * TILE_SIZE] = make_dead_end_right()
            elif maze[x, y] == 0:
                pass

    plt.imshow(maze_image, cmap='Blues_r', origin='lower')
    plt.xticks([]), plt.yticks([])
    plt.show()

def test_plot_maze():
    size = MAZE_SIZE

    maze = np.zeros((size, size))

    tiles = iter([
        VERTICAL, HORIZONTAL, CORNER_L, CORNER_REVERSE_L, CORNER_7, CORNER_REVERSE_7,
        JUNCTION_T, JUNCTION_T_90, JUNCTION_T_180, JUNCTION_T_270, JUNCTION_CROSS,
        DEAD_END_UP, DEAD_END_DOWN, DEAD_END_LEFT, DEAD_END_RIGHT
    ])

    for x in range(size):
        for y in range(size):
            try:
                maze[x, y] = next(tiles)
            except StopIteration:
                maze[x, y] = 0

    plot_maze(maze)
    exit()

def get_action_vec(maze, cur_pos, heading):
    """
    based on the direction the agent is facing, can it go left, straight, or right
    """
    actions = np.zeros(3) # left, straight, right

    up_pos = add_pos(cur_pos, UP)
    up_avail = False
    if within_bounds(up_pos, maze.shape):
        if check_up(maze[up_pos[0], up_pos[1]]):
            up_avail = True

    down_pos = add_pos(cur_pos, DOWN)
    down_avail = False
    if within_bounds(down_pos, maze.shape):
        if check_down(maze[down_pos[0], down_pos[1]]):
            down_avail = True

    left_pos = add_pos(cur_pos, LEFT)
    left_avail = False
    if within_bounds(left_pos, maze.shape):
        if check_left(maze[left_pos[0], left_pos[1]]):
            left_avail = True

    right_pos = add_pos(cur_pos, RIGHT)
    right_avail = False
    if within_bounds(right_pos, maze.shape):
        if check_right(maze[right_pos[0], right_pos[1]]):
            right_avail = True

    if heading == UP:
        if left_avail:
            actions[0] = 1
        if up_avail:
            actions[1] = 1
        if right_avail:
            actions[2] = 1
    elif heading == DOWN:
        if right_avail:
            actions[0] = 1
        if down_avail:
            actions[1] = 1
        if left_avail:
            actions[2] = 1
    elif heading == LEFT:
        if down_avail:
            actions[0] = 1
        if left_avail:
            actions[1] = 1
        if up_avail:
            actions[2] = 1
    elif heading == RIGHT:
        if up_avail:
            actions[0] = 1
        if right_avail:
            actions[1] = 1
        if down_avail:
            actions[2] = 1

    return actions

def action_to_dir(action, heading):
    if heading == UP:
        if action == 0:
            return LEFT
        elif action == 1:
            return UP
        elif action == 2:
            return RIGHT
    elif heading == DOWN:
        if action == 0:
            return RIGHT
        elif action == 1:
            return DOWN
        elif action == 2:
            return LEFT
    elif heading == LEFT:
        if action == 0:
            return DOWN
        elif action == 1:
            return LEFT
        elif action == 2:
            return UP
    elif heading == RIGHT:
        if action == 0:
            return UP
        elif action == 1:
            return RIGHT
        elif action == 2:
            return DOWN

def actions_to_dirs(action_vec, heading):
    return {action_to_dir(action, heading) for action in np.flatnonzero(action_vec)}

def update_dir(maze, cur_pos, cur_dir):
    if maze[cur_pos[0], cur_pos[1]] == DEAD_END_UP:
        return UP
    elif maze[cur_pos[0], cur_pos[1]] == DEAD_END_DOWN:
        return DOWN
    elif maze[cur_pos[0], cur_pos[1]] == DEAD_END_LEFT:
        return LEFT
    elif maze[cur_pos[0], cur_pos[1]] == DEAD_END_RIGHT:
        return RIGHT

    return cur_dir

if __name__ == "__main__":
    debug = False
    desired_junction_ratio = 0.05 # percentage of paths that have 3 or 4 exits

    # test_plot_maze()

    maze, start, start_dir = create_maze(desired_junction_ratio, debug=debug)

    junction_ratio = get_maze_junction_ratio(maze)
    print(f"Junction ratio: {junction_ratio}")

    cur_pos = start
    cur_dir = start_dir
    for _ in range(1000):
        action_vec = get_action_vec(maze, cur_pos, cur_dir)
        print(f'cur_pos: {cur_pos}')
        print(f'heading: {dir_str(cur_dir)}')
        print(f'action vec: {action_vec}')
        actions = actions_to_dirs(action_vec, cur_dir)
        print(f'actions: {dirs_str(actions)}')
        plot_maze(maze, cur_pos, debug=True)
        action = random.choice(list(actions))
        print(f'Action: {dir_str(action)}')
        cur_pos = add_pos(cur_pos, action)
        cur_dir = update_dir(maze, cur_pos, action)
