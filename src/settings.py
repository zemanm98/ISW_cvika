import datetime
import logging
import numpy as np

WIDTH = 400
HEIGHT = 600
ENVSPACE = 400
STATUSSPACE = 40
MARGIN = 20
BORDER = 3
STATUS_BORDER = 1
RENDER_PRECISION = 4

ACTIONS_X1 = BORDER + MARGIN
ACTIONS_Y1 = ENVSPACE + BORDER + MARGIN
ACTIONS_HISTOGRAM_MARGIN_RATIO = 0.2

ENV_X1 = BORDER + MARGIN
ENV_Y1 = BORDER + MARGIN

STATUS_X1 = BORDER + MARGIN
STATUS_Y1 = ENVSPACE - MARGIN - STATUSSPACE

ENV_WIDTH = WIDTH - 2*(BORDER + MARGIN)
ENV_HEIGHT = ENVSPACE - BORDER - 2*MARGIN - STATUSSPACE
STATUS_WIDTH = WIDTH - 2*(BORDER + MARGIN)
STATUS_HEIGHT = STATUSSPACE
ACTIONS_WIDTH = WIDTH - 2*(BORDER + MARGIN)
ACTIONS_HEIGHT = HEIGHT - ENVSPACE - 2*(BORDER + MARGIN)
ACTIONS_HIST_HEIGHT = ACTIONS_HEIGHT * 0.8
HIST_MARGIN = 50
HIST_HEIGHT = 0.5

# TOURNAMENTS
MAX_STABLE_VERSIONS_IN_GROUP = 9
ROUNDS_TOTAL = 5

# LOGGING
VERSIONS_PATH = "versions/"
LOG_FILE = 'logs/' + datetime.datetime.now().strftime("%Y-%b-%d_%H-%M-%S") + '.log'
LOG_ROOT = 'logs'

LOGGER_FORMAT = '[ %(levelname)7s %(asctime)s %(filename)s:line %(lineno)4s - %(funcName)20s() ] %(message)s'
logger = logging.getLogger("prop_agents")
formatter = logging.Formatter(LOGGER_FORMAT)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)
np.set_printoptions(precision=3, suppress=True, linewidth=np.inf, threshold=np.inf)
logger.setLevel(logging.CRITICAL)
