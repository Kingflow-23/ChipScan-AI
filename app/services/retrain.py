import logging
from utils.training_yolo import train_model

logger = logging.getLogger(__name__)


def start_retraining(retrain: bool = True, retraining_status: dict = None):
    """
    Runs YOLO retraining synchronously.
    Threading is handled by the Flask route.
    """
    logger.info("Training started")
    train_model(retrain=retrain, status_dict=retraining_status)
    logger.info("Training finished")
