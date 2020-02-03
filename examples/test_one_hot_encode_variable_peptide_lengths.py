import pace, pace.sklearn
import pickle
import copy

with open('/home/dcraft/ImmunoOncology/pace/x.pickle', 'rb') as f:
    x = pickle.load(f)

#print(x)

# padded version
xp = copy.deepcopy(x)

for i in range(len(x)):
    for j in range(11 - len(x[i])):
        xp[i].append('A')

encoder = pace.sklearn.create_one_hot_encoder(len(xp[0]))
encoder.fit(xp)
xenc = encoder.transform(xp).toarray()

xencstrip = [[] for i in range(len(x))]

#now strip off the padding
for i in range(len(x)):
    if len(x[i]) < 11:
        xencstrip[i] = xenc[i, :(-(11 - len(x[i])) * 20)]
    else:
        xencstrip[i] = xenc[i, :]

#check
for i in range(len(x)):
    print(len(x[i]))
    if len(x[i]) == 8:
        print('8: ' + str(len(xencstrip[i])))
    elif len(x[i]) == 9:
        print('9: ' + str(len(xencstrip[i])))
    elif len(x[i]) == 10:
        print('10: ' + str(len(xencstrip[i])))
    else:
        print('11: ' + str(len(xencstrip[i])))
