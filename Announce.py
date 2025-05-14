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


def set_logger(args):
    if not os.path.exists(args['save_dir']):
        os.makedirs(os.path.join(os.getcwd(), args['save_dir']))
    log_file = os.path.join(args['save_dir'], 'log_'+args['save_file'][:-4]+'.txt')    
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)