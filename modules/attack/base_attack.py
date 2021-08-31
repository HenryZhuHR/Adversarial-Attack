import os
from abc import abstractmethod, ABCMeta
from typing import List
from torch import Tensor, nn


class BaseAttack(metaclass=ABCMeta):
    attack_name = str()
    ATTACK_PARAMETERS = list()  # all attack parameter needed for this attack method

    def __init__(self, model, device='cpu', **kw_attack_params):
        self.model:nn.Module = model
        self.device:str = device

    @abstractmethod
    def attack(self,
               input,
               label
               ):
        """
        attack
        ===        
        """
        attack_input = input
        return attack_input

    def parse_params(self, kwargs: dict) -> dict:
        """
        Parse Attack Parameters
        ===
        parse attack parameters according to attack method
        """
        print('%s%sGet attack parameters( √ means valid param) :' %
              ('-'*33, os.linesep))
        for key in kwargs.keys():
            print('  %s %s: %f' %
                  ('√' if key in self.ATTACK_PARAMETERS else '-', key, kwargs[key]))

        return {key: value for key, value in kwargs.items() if key in self.ATTACK_PARAMETERS}
