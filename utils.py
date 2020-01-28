import numpy as np
import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
    
def padding(input_ids,MAX_LEN=128,padding_value=0):
    '''pad a list of input ids to have fixed max length, truncating if necessary'''
    for sen in input_ids:
        if len(sen)>MAX_LEN:
            sen[:] = sen[:MAX_LEN]
        else:
            sen[:] = sen + [padding_value]*(MAX_LEN-len(sen))
    input_ids =np.int64(input_ids)
    return input_ids