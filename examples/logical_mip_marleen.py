import pace, pace.sklearn, pace.featurization
import sklearn.linear_model
import pprint
import sklearn


class MarleenAlg(pace.PredictionAlgorithm):
    def train(self, binders, nonbinders):
        x = [list(s.peptide)
             for s in binders] + [list(s.peptide) for s in nonbinders]
        y = [1] * len(binders) + [0] * len(nonbinders)

        x = pace.featurization.do_FMLN_encoding(x,m=4,n=5)

        encoder = pace.sklearn.create_one_hot_encoder(len(x[0]))
        encoder.fit(x)
        encoded_x = encoder.transform(x).toarray()
        # encoded_x is n samples by 180 features and y is binary
        #call my mip optimizer
        self.clf = sklearn.linear_model.RidgeClassifier().fit(encoded_x, y)

    def predict(self, samples):
        x = [list(s.peptide) for s in samples]

        x = pace.featurization.do_FMLN_encoding(x,m=6,n=5)

        encoder = pace.sklearn.create_one_hot_encoder(len(x[0]))
        encoder.fit(x)
        encoded_x = encoder.transform(x).toarray()

        return self.clf.predict(encoded_x)



a02 = ['A0203']
a68 = ['A6802']
b35 = ['B3501']
#my_scorers = {'ppv': pace.evaluation.PpvScorer(), 'accuracy': pace.evaluation.AccuracyScorer(cutoff=0.6)}

#note setting nbr_train to 10 improves things by a couple points.
scores, all_fold_results = pace.evaluate(lambda: ExploreAlgorithm(5,4,encoding_style='one_hot'), selected_lengths=[9],
                                         selected_alleles=b35, dataset=pace.data.load_dataset(16), nbr_train=1, nbr_test=10)

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