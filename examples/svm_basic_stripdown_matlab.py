import pace, pace.sklearn, pace.featurization
import pprint
import pandas as pd 
import numpy as np 
from sklearn import svm

class svm_basic_stripdown_matlab():
    def write(self, hits, misses):
        x = [list(s.peptide) for s in hits] + [list(s.peptide) for s in misses]
        xa = [s.allele for s in hits] + [s.allele for s in misses]
        
        y = [1] * len(hits) + [0] * len(misses)

        # all
        fmln_x = pace.featurization.do_FMLN_encoding(x)
        encoded_x = pace.featurization.do_5d_encoding(fmln_x)
        
        #write data or send to matlab



whichAlleleSet=16
