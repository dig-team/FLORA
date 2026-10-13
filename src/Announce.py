"""
This file is part of FLORA, an unsupervised system for automatic knowledge graph (KG) alignment. 
The file is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0) by Yiwen Peng, Thomas Bonald, Fabian Suchanek and Lingyun Huang.

Description: Lightweight console and logging helpers for reporting progress during FLORA runs.
"""

import os
import logging

indentation=0
spaces="                      "
isDoing=False

def doing(*message):
    global isDoing
    global spaces
    global indentation
    if isDoing:
        print()
    print(spaces[0:indentation*2], end='')
    for m in message:
        print(m, end='')
        print(' ', end='')
    print("... ", end='', flush=True)
    indentation+=1
    isDoing=True    

def done(*message):
    global isDoing
    global spaces
    global indentation
    indentation-=1
    if not isDoing:        
        print(spaces[0:indentation*2], end='') 
    if len(message):
        print("done (", end='')
        for m in message:
            print(m, end='')
            print(' ', end='')
        print(")", flush=True)    
    else:
        print("done", flush=True)
    isDoing=False

def message(*message):
    global isDoing
    if isDoing:
        print()
    print(' '.join(str(m) for m in message), flush=True)
    isDoing=False

def set_logger(args):
    log_dir = '../save/logs'
    os.makedirs(log_dir, exist_ok=True)
    output_stem = os.path.splitext(os.path.basename(args['output']))[0]
    log_file = os.path.join(log_dir, 'log_'+output_stem+'.txt')
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # logging.getLogger('').addHandler(console)