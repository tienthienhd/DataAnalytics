# -*- coding: utf-8 -*-

import json

class Config(object):
    def __init__(self, filepath):
        with open(filepath, 'r') as config_file:
            self.config_dict = json.load(config_file)
            
    def get_config(self):
        return self.config_dict