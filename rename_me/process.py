import logging
from typing import List

"""Code and objects used for predicting or scoring"""

logger = logging.getLogger(__name__)


class Processor:
    def process(self, input: List):
        logger.info('Processing...')
        raise NotImplementedError
