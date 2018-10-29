# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 09:44:16 2018

@author: Administrator
"""

import time

def tic():
    globals()['tt'] = time.clock()

def toc():
    print('\nElapsed time: %.8f seconds\n' % (time.clock()-globals()['tt']))