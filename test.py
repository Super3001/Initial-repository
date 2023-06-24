from sklearn.model_selection import KFold
import numpy

kfold = KFold(n_splits=20, shuffle=True)

arr = numpy.arange(0,40)

for fold, (train, val) in enumerate(kfold.split(arr)):
    print(f'fold{fold}:')
    print(train)
    print(val)