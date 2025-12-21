import logging
from utils.training_yolo import train_model

logger = logging.getLogger(__name__)


def start_retraining(resume: bool = True):
    """
    Runs YOLO retraining synchronously.
    Threading is handled by the Flask route.
    """
    logger.info("Training started")
    train_model(resume=resume)
    logger.info("Training finished")
