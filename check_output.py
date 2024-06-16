#  Script to help check for any changes in results after "cosmetic" or "cleanup" work

import numpy as np
import pandas as pd
from common_params import *
stringency = 0.0001

def max_frac_diff(f_ref, f_test, thresh):

    # Load values -- customized depending on type
    data_ref = pd.csvread(f_ref)
    data_test = pd.csvread(f_test)

    #  Calculate maximum % difference in values
    max_diff = np.max(np.abs(np.divide(np.subtract(data_test, data_ref), data_ref)))
    return(max_diff>thresh)


# set up list of files
file = 'RampRposSGradual80_fitResults_combined.csv'
ref =  'MainPaperData/'
ref_file = FWDOUTPUTDIR + ref + file
test_file = FWDOUTPUTDIR + file + 'file2.csv'

max_diff = max_frac_diff(ref_file, test_file)
if max_diff > stringency:
    print('ALERT! Detected a percentage difference of: ', max_diff*100, '%')
    # then indicate where the differences are
else:
    print('Excellent! No differences above threshold were found.')

