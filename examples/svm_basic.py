import pace, pace.sklearn, pace.featurization
import pprint
import pandas as pd 
import numpy as np 
from sklearn import svm
import scipy.io
import random

class svm_basic(pace.PredictionAlgorithm):
    def train(self, hits, misses):
        x = [list(s.peptide) for s in hits] + [list(s.peptide) for s in misses]
        xa = [s.allele for s in hits] + [s.allele for s in misses]
        
        y = [1] * len(hits) + [0] * len(misses)

        # all
        fmln_x = pace.featurization.do_FMLN_encoding(x)
        encoded_x = pace.featurization.do_5d_encoding(fmln_x)
        
        def my_kernel(X1, X2):
            """
            Kernel function takes two inputs: X1 and X2. X1 is one 
            set of samples (num samples 1 x num features), and X2 is another set of samples.
            What gets returned is a similarity matrix size (num samples 1 x num samples 2)
            
            A row [of either X1 or X2] consists of:
            HLA name, peptide string [uniform length e.g. FMLN encoding, at least for now]
            """

            #to test just return random symmetric matrix
            s=X1.shape
            N = s[0]
            b = np.random.random(size=(N,N))
            b_symm = (b + b.T)/2
            np.fill_diagonal(b_symm, 1.0)
            return b_symm

        # prepare and write sample matrix for my_kernel
        Xsvm = pd.DataFrame({'HLA':xa, 'peptide':fmln_x, 'y':y})
        scipy.io.savemat('trainDataSVM.mat', {'struct1':Xsvm.to_dict("list")})

        #Xsvm = np.random.random(size=(len(y),4))
        
        #self.clf = svm.SVC(kernel=my_kernel)
        #self.clf.fit(Xsvm, y)
        # self.clf = sklearn.linear_model.RidgeClassifier(alpha=1.0).fit(encoded_x, y)


    # pure guessing
    def predict(self, samples):
        return [random.uniform(0, 1) for _ in samples]

whichAlleleSet=16

scores = pace.evaluate(svm_basic,
                       selected_lengths=[8,9,10,11],dataset=pace.data.load_dataset(whichAlleleSet))
pprint.pprint(scores)
