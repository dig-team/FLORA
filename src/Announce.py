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
    if not os.path.exists('../save/logs/'):
        os.makedirs(os.path.join(os.getcwd(), '../save/logs/'))
    log_file = os.path.join('../save/logs/', 'log_'+args['output'][:-4]+'.txt')    
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