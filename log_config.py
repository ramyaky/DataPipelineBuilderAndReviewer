import logging
import os

def setup_logging(name="DataPipelineBuilder", log_level=logging.INFO, log_file="app.log"):
    #Create logs directory if not exists
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", log_file)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, "a"),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(name)
    logger.info("Logging Initialized...")
    return logger