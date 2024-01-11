import matplotlib.pyplot as plt
import numpy as np

class Board: 
    def __init__(self) -> None:
        self.maze = [
                        [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
                        [ 1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.],
                        [ 1.,  1.,  1.,  1.,  0.,  1.,  0.,  1.],
                        [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
                        [ 1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
                        [ 1.,  1.,  1.,  0.,  1.,  0.,  0.,  0.],
                        [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
                        [ 1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.]
                    ]
        self.initial_position = (0,0)
        self.exit = (7,7)

    def show(self, rat):
        plt.grid('on')
        nrows = 8
        ncols = 8
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, nrows, 1))
        ax.set_yticks(np.arange(0.5, ncols, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        canvas = np.copy(self.maze)

        canvas[rat[0], rat[1]] = 0.3   # rat cell
        canvas[nrows-1, ncols-1] = 0.9 # cheese cell
        plt.imshow(canvas, interpolation='none', cmap='afmhot')
        plt.show()

        return None
    
if __name__ == '__main__':
    board = Board()
    board.show((0,0))