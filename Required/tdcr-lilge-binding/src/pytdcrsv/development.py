import logging

logger = logging.getLogger(__name__)
CLEVEL = logging.DEBUG
FLEVEL = logging.DEBUG
FNAME = 'pytdrcsv.log'
USELOGFILE = True
TESTING = False

# Create handlers
C_HANDLER = logging.StreamHandler()
C_HANDLER.setLevel(CLEVEL)

# Create formatters and add it to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

# These are passed to the logger
C_HANDLER.setFormatter(c_format)

# Same as before, but for a file logger
F_HANDLER = logging.FileHandler(FNAME)
F_HANDLER.setLevel(FLEVEL)
f_format = logging.Formatter('%(asctime)s - '
                             '%(name)s - '
                             '%(levelname)s - '
                             '%(message)s')
F_HANDLER.setFormatter(f_format)
logging.basicConfig(filename=FNAME, level=CLEVEL)
logger.setLevel(CLEVEL)