# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 09:52:45 2018

@author: Administrator
"""

from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello Flask!'


if __name__ == '__main__':
    app.run()