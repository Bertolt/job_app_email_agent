import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Root logger configuration
logger = logging.getLogger(__name__)


def configure_logging(agent_name=None):
    """
    Configure logging with daily log files.
    Args:
        agent_name: Optional agent name to include in the log filename
    Returns:
        str: Path to the configured log file
    """
    daily_timestamp = datetime.now().strftime("%Y%m%d")
    # Create filename with optional agent name
    filename_parts = []
    if agent_name:
        # Clean the agent name for use in filenames
        clean_name = "".join(c if c.isalnum() else "_" for c in agent_name)
        filename_parts.append(clean_name)
    filename_parts.append(daily_timestamp)

    log_filename = os.path.join(LOG_DIR, f"Agent_{'_'.join(filename_parts)}.log")

    # Remove any existing handlers to avoid duplicate logs
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    # Configure the logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(),  # Also output to console
        ],
        force=True,  # Override any existing configuration
    )

    return log_filename


class AgentLogger:
    """
    Custom logger class that provides logging methods with agent context.
    This allows for a more intuitive syntax like self.log.error("message").
    """

    def __init__(self, agent):
        self.agent = agent

    def _log(self, level, message, *args):
        color_code = self.agent.BG_BLACK + self.agent.color

        # Format the message with args if provided
        if args:
            message = message % args

        formatted_message = "[%s] %s" % (self.agent.name, message)

        # Get the logger method corresponding to the level
        log_method = getattr(logger, level)
        log_method("%s%s%s", color_code, formatted_message, self.agent.RESET)

    def debug(self, message, *args):
        self._log("debug", message, *args)

    def info(self, message, *args):
        self._log("info", message, *args)

    def warning(self, message, *args):
        self._log("warning", message, *args)

    def error(self, message, *args):
        self._log("error", message, *args)

    def critical(self, message, *args):
        self._log("critical", message, *args)


class Agent:
    """
    Abstract base class for agents that perform specific tasks.
    This class provides a foundation for creating various types of agents
    that can process and handle different operations.
    Used to log messages in a way that can identify each Agent
    """

    # Foreground colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Background color
    BG_BLACK = "\033[40m"

    # Reset code to return to default color
    RESET = "\033[0m"

    name: str = ""
    color: str = "\033[37m"

    def __init__(self):
        """
        Initialize the agent instance.
        """
        # create the output directory where the outputs will be saved by default
        self.output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_file = configure_logging(self.name)
        self._logger = AgentLogger(self)

    @property
    def log(self):
        """
        Access the agent logger object.

        Returns:
            AgentLogger: A logger object with methods for each severity level
        """
        return self._logger

    # Keep the original log method for backward compatibility
    def log_message(self, message, *args, level="info"):
        """
        Log a message with the specified severity level, identifying the agent.

        Args:
            message: The message to log (may contain formatting placeholders)
            *args: Arguments for message formatting
            level: Severity level (debug, info, warning, error, critical)
        """
        getattr(self.log, level.lower())(message, *args)
