# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 04:17:14 2020

@author: User
"""

import os

dirname = os.path.realpath(__file__)
dirname[:-len("mego_all_in_one.py")]
filepath = os.path.join(dirname, ".\", "report")
os.chdir(filepath)