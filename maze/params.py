
# --------------------- agent.py --------------------- #
GAMES = 1000
MAX_MEM = 100_000
EPSILON_INIT = 1. # exploration init
EPSILON_DEC = EPSILON_INIT/GAMES # exploration decrement
ALPHA = .01 # learning rate
GAMMA = .9 # discount rate
BATCH_SIZE = 1000
FLAT_SIZE = 64
CHOICES = 4 # for now we are using basic maze choices for simplicity
HIDDEN_SIZE = 256
DIM = 8

# ---------------------- env.py ---------------------- #

# --------------------- model.py --------------------- #
