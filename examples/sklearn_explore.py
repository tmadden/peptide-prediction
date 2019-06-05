import pace, pace.sklearn, pace.featurization, pace.evaluation
import sklearn.linear_model
import sklearn.gaussian_process
from sklearn.gaussian_process.kernels import RBF
import pprint
import sklearn
import calendar, time
import pandas as pd 
import scipy.io
import numpy as np
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from Bio import motifs
from Bio.Alphabet import IUPAC
from Bio.Seq import Seq

class ExploreAlgorithm(pace.PredictionAlgorithm):

    def __init__(self, fmln_m, fmln_n, encoding_style='one_hot'):
        self.fmln_m = fmln_m
        self.fmln_n = fmln_n
        self.encoding_style = encoding_style

    def do_encoding(self,x):
        xorig = x
        x = pace.featurization.do_FMLN_encoding(x,m=self.fmln_m,n=self.fmln_n)
        if self.encoding_style == 'one_hot':
            encoder = pace.sklearn.create_one_hot_encoder(len(x[0]))
            encoder.fit(x)
            r = encoder.transform(x).toarray()
            #also do binary encoding of peptide length
            bepl = pace.featurization.do_binary_peptide_length_encoding(xorig)
            return np.concatenate((r,bepl),axis=1)
            #return r
        elif self.encoding_style == '5d':
            return pace.featurization.do_5d_encoding(x)
        else:
            raise Exception('Unknown encoding style used: '+self.encoding_style)
        
    def train(self, binders, nonbinders):
        #to replicate the binders:
        #rep_binders = binders*10 --> then replace binders in next two lines with rep_binders.
        x = [list(s.peptide)
             for s in binders] + [list(s.peptide) for s in nonbinders]
        y = [1] * len(binders) + [0] * len(nonbinders)

        encoded_x = self.do_encoding(x)

        #create forbidden residues rules: look at binders and form something like:
        # e.g. for B*35:01
        #forbidden_residues = [['C','K'],['C','E','F','H','K','Q','R','W','Y'],None,None,
        #                      ['R'],None,['C'],None,['A','C','D','E','G','K','N','P','Q','R']]
        binding_peptide_list = [s.peptide for s in binders]
        instances = [Seq(s,IUPAC.protein) for s in binding_peptide_list]
        m = motifs.create(instances,alphabet=IUPAC.protein)

        self.forbidden_residues=[]
        numaa = len(m.counts) #20 amino acids
        pep_length = len(m) #9, length of peptide
        pep_pos_res = [m.counts[:,i] for i in range(pep_length)]
        #e.g. pep_pos_res[0] gives the count,of how many times each amino acid appears at the first position, as a dictionary

        for i in range(pep_length):
            self.forbidden_residues.append([])
            for res, cnt in pep_pos_res[i].items():
                if cnt == 0:
                    self.forbidden_residues[i].append(res)

        print('during this training round, forbidden residues are')
        print(self.forbidden_residues)

        #encoded_x = pace.featurization.do_5d_encoding(x)

        #HOW TO SEND DATA TO MATLAB AND TRAIN THERE, without a process hook.

        #get a time stamp and save data with it:
        #ts=calendar.timegm(time.gmtime())-1559000000
        #scipy.io.savemat('trainData'+str(ts)+'.mat', {'x':encoded_x,'y':y})
        
        #this works if we ever need to call matlab. but it's slow since matlab startup is slow.
        #process handle for my version of matlab only currently works for python3.5 and lower. erg. andy kaplan call??
        #cmd = '''/usr/local/MATLAB/R2016b/bin/matlab -nodesktop -nosplash -r "myTrainFunction('trainData150503.mat');exit;"'''
        #print(cmd)
        #os.system(cmd)

        #self.clf = sklearn.linear_model.RidgeClassifier().fit(encoded_x, y)
        #SGD is better: either with default loss [which makes a linear svm] or log loss. using log here for adaboost below. 
        #self.clf = sklearn.linear_model.SGDClassifier(loss='log',max_iter=800,tol=1e-2,penalty='l2').fit(encoded_x, y)
        #self.clf = sklearn.neighbors.KNeighborsClassifier(10, weights='distance').fit(encoded_x, y)
        #RBF SVM is best yet tested
        #self.clf = sklearn.svm.SVC(C=10,kernel='rbf',gamma='scale').fit(encoded_x, y)
        #linear also good.
        #self.clf = sklearn.svm.SVC(C=.1,kernel='linear').fit(encoded_x, y)
        
        #try Gaussian Naive Bayes. not so good.
        #self.clf = GaussianNB().fit(encoded_x, y)
        
        #this one: gaussian process, also does well but is very slow.
        #kernel = 1.0 * RBF(1.0)
        #self.clf = sklearn.gaussian_process.GaussianProcessClassifier(kernel=kernel).fit(encoded_x, y)

        #try adaboost
        #first with default base learner, a decision tree. does pretty good. max_depth=1 seems the best
        #self.clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),n_estimators=100).fit(encoded_x, y)

        #adaboost doesn't seem to help the SGD classifier, at least not this one:
        #self.clf = AdaBoostClassifier(base_estimator=sklearn.linear_model.SGDClassifier(loss='log',max_iter=800,tol=1e-2,penalty='l2'),n_estimators=100).fit(encoded_x, y)

        #not so good:
        #self.clf = AdaBoostClassifier(base_estimator=GaussianNB(),n_estimators=4).fit(encoded_x, y)
        
        #not so good
        #self.clf = QuadraticDiscriminantAnalysis()
        #self.clf.fit(encoded_x, y)

        #self.clf = LinearDiscriminantAnalysis()
        #self.clf.fit(encoded_x, y)

        #ok try sklearn's random forest. pretty good.
        #self.clf = RandomForestClassifier(n_estimators=50, max_depth=None)
        #self.clf.fit(encoded_x, y)

        #ok now try voting classifier using a bunch (can take or leave the sgd..): this here is a good reasonable set. below will try adding in more.
        
        self.clf = VotingClassifier(estimators=[
            ('svmrbf', sklearn.svm.SVC(C=10,kernel='rbf',gamma='scale', probability=True)), 
            ('svmlin', sklearn.svm.SVC(C=.1,kernel='linear', probability=True)), 
            ('rf', RandomForestClassifier(n_estimators=30, max_depth=None))], voting='soft')
        self.clf.fit(encoded_x, y)
        
        
        '''
        self.clf = VotingClassifier(estimators=[
            ('adaboost', AdaBoostClassifier()),
            ('svmrbf', sklearn.svm.SVC(C=10,kernel='rbf',gamma='scale', probability=True)), 
            ('svmlin', sklearn.svm.SVC(C=.1,kernel='linear', probability=True)), 
            ('rf', RandomForestClassifier(n_estimators=50, max_depth=None)), 
            ('sgd', sklearn.linear_model.SGDClassifier(loss='log',max_iter=800,tol=1e-2,penalty='l2'))], voting='soft')
        self.clf.fit(encoded_x, y)
        '''

    def predict(self, samples):
        x = [list(s.peptide) for s in samples]
        encoded_x = self.do_encoding(x)
        #to get probabilities rather than 0/1, use predict_proba
        #but note this gives a probability for each class so in our binary case
        #it gives an n_samples x 2 array. just use second column, which is prob of "1", binder.
        #does much better for ppv!
        r = self.clf.predict_proba(encoded_x)

        # manually adjust probabilities according to hard coded forbidden residues rule [which we would learn from training data]
        # hard coding for B*35:01
        #forbidden_residues = [['C','K'],['C','E','F','H','K','Q','R','W','Y'],None,None,
        #                      ['R'],None,['C'],None,['A','C','D','E','G','K','N','P','Q','R']]

        def has_forbidden_residue(myseq,forbidden_residues):
            itisok=True
            for c, fr in zip(list(myseq),forbidden_residues):
                if fr==None:
                    continue
                else:
                    for j in fr:
                        if j==c:
                            itisok = False
                            break
            return not itisok
        
        for i in range(len(samples)):
            if has_forbidden_residue(samples[i].peptide,self.forbidden_residues):
                # print('changing probability from '+str(r[i])+' to 0.001 for sequence: '+samples[i].peptide)
                r[i] = [0.999, 0.001]
        

        return r[:,1]
        #return self.clf.predict(encoded_x)

a02 = ['A0203']
a68 = ['A6802']
b35 = ['B3501']
#my_scorers = {'ppv': pace.evaluation.PpvScorer(), 'accuracy': pace.evaluation.AccuracyScorer(cutoff=0.6)}

#note setting nbr_train to 10 improves things by a couple points.
scores, all_fold_results = pace.evaluate(lambda: ExploreAlgorithm(5,4,encoding_style='one_hot'), selected_lengths=[9],
                                         selected_alleles=b35, dataset=pace.data.load_dataset(16), nbr_train=1, nbr_test=100)

pprint.pprint(scores)
ppvvals = scores["ppv"]
print('mean ppv = '+str(np.mean(ppvvals)))
print('std ppv = '+str(np.std(ppvvals)))

combined_ppv = pace.evaluation.score_by_ppv([r.truth for r in all_fold_results],
                                 [r.prediction for r in all_fold_results])
print('combined ppv = '+str(combined_ppv))
#loop to get all the solo allele results for length 9
#s9 = [pace.evaluate(lambda: ExploreAlgorithm(4,5,encoding_style='one_hot'), selected_lengths=[9],
#                       selected_alleles=[a], dataset=pace.data.load_dataset(16)) for a in pace.featurization.a16_names]

# double loop, lengths and alleles
#this way below works but hard to print out stuff along way / debug so...
#r = [ [pace.evaluate(lambda: ExploreAlgorithm(4,4+i,encoding_style='one_hot'), selected_lengths=[i+8],
#                    selected_alleles=[a], dataset=pace.data.load_dataset(16)) for i in range(4) ] 
#                    for a in pace.featurization.a16_names]

#do this way instead.
#forget the 8 mers since a couple have tiny sample sizes
'''
r=[[] for i in range(3)]
for a in pace.featurization.a16_names:
    for i in range(3):
        print('************************      running allele '+a+' length '+str(9+i))
        myr = pace.evaluate(lambda: ExploreAlgorithm(5,4+i,encoding_style='one_hot'), selected_lengths=[i+9],
                            selected_alleles=[a], dataset=pace.data.load_dataset(16))
        r[i].append(myr)
scipy.io.savemat('newResultsPROBA.mat', mdict={'r': r})
'''