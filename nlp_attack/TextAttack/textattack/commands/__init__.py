"""

TextAttack commands Package
===========================

"""
import tensorflow as tf     
tf.config.set_visible_devices([], 'GPU')    

from abc import ABC, abstractmethod
from .textattack_command import TextAttackCommand
from . import textattack_cli
