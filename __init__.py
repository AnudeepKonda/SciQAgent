import logging


def setup_logger():
    """
    Setup a logger for the rag agent.

    This function configures a logger with a specific format and level,
    and ensures that duplicate handlers are not added.
    """
    # Create a custom logger
    logger = logging.getLogger('rag_agent')

    # Set the level of the logger to INFO
    logger.setLevel(logging.INFO)

    # Check if the logger has handlers already to avoid adding multiple handlers
    if not logger.handlers:
        # Create handlers
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatters and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(console_handler)

    # Disable logging from libraries
    logging.getLogger('external_library').setLevel(logging.WARNING)
