import logging

# Configure the logger
logging.basicConfig(level=logging.DEBUG)

# Create a logger
logger = logging.getLogger(__name__)

# Create a console handler and set the level to DEBUG
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a formatter and attach it to the console handler
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)
