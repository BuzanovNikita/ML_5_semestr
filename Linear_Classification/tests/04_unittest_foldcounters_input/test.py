import numpy as np
import pandas as pd

from Task import MyOneHotEncoder, SimpleCounterEncoder, FoldCounters, weights

def test_imports():
    with open('Task.py', 'r') as file:
        lines = ' '.join(file.readlines())
        assert 'import numpy' in lines
        assert lines.count('import') == 1
        assert 'sklearn' not in lines
        assert 'get_dummies' not in lines

def test_kfold_counters_small():
    data = {'col_1': [0,1,0,1,0,1,0,1,0,1,0,1], 'col_2':['a','b','c','a','b','c','a','b','c','a','b','c'], 'col_3': [1,2,3,4,1,2,3,4,1,2,3,4]}
    df_test = pd.DataFrame.from_dict(data)
    enc = FoldCounters(n_folds=2)
    enc.fit(df_test[['col_1', 'col_2']], df_test['col_3'], seed=6)
    counts = enc.transform(df_test[['col_1', 'col_2']], a=0, b=0)
    ans = np.array([[7/3,0.5,14/3,3,1/3,9],\
                    [8/3,0.5,16/3,2,1/3,6],\
                    [5/3,0.5,10/3,2.5,1/3,7.5],\
                    [10/3,0.5,20/3,2,1/3,6],\
                    [5/3,0.5,10/3,3,1/3,9],\
                    [10/3,0.5,20/3,2.5,1/3,7.5],\
                    [7/3,0.5,14/3,3,1/3,9],\
                    [8/3,0.5,16/3,2,1/3,6],\
                    [7/3,0.5,14/3,2.5,1/3,7.5],\
                    [10/3,0.5,20/3,2,1/3,6],\
                    [5/3,0.5,10/3,3,1/3,9],\
                    [8/3,0.5,16/3,2.5,1/3,7.5]])
    assert len(counts.shape) == 2
    assert counts.shape[0] == 12
    assert counts.shape[1] == 6
    assert np.allclose(counts, ans, atol=1e-8)
    assert type(counts) == np.ndarray

def test_kfold_counters_diffshape_idx_small():
    data = np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]])
    index = pd.Index([11, 0, 9, 7, 6, 10, 1, 2, 5, 4, 3, 8])
    df_test = pd.DataFrame(data.reshape(12, 2), index=index, columns=['col_1', 'col_2'])
    enc = FoldCounters(n_folds=2)
    enc.fit(df_test[['col_1']], df_test['col_2'], seed=6)
    counts = enc.transform(df_test[['col_1']], a=0, b=0)
    ans = np.array([[ 1.        ,  0.66666667,  1.5       ],\
       [ 1.        ,  0.66666667,  1.5       ],\
       [ 1.        ,  0.33333333,  3.        ],\
       [ 1.        ,  0.33333333,  3.        ],\
       [ 1.        ,  0.33333333,  3.        ],\
       [ 1.        ,  0.33333333,  3.        ],\
       [ 2.        ,  0.16666667, 12.        ],\
       [ 4.        ,  0.16666667, 24.        ],\
       [ 2.        ,  0.16666667, 12.        ],\
       [ 4.        ,  0.33333333, 12.        ],\
       [ 2.        ,  0.33333333,  6.        ],\
       [ 4.        ,  0.16666667, 24.        ]])
    assert len(counts.shape) == 2
    assert counts.shape[0] == 12
    assert counts.shape[1] == 3
    assert np.allclose(counts, ans, atol=1e-8)
    assert type(counts) == np.ndarray

def test_fold_counters_big():
    data = {'col_1': [1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 2, 1, 2, 0, 2, 1, 2, 0, 0, 2, 0, 1, 2, 2, 0, 1, 1, 2, 0], 'col_2': [1, 1, 1, 1, 0, 4, 1, 0, 0, 3, 2, 1, 0, 3, 1, 1, 3, 4, 0, 1, 3, 4, 2, 4, 0, 3, 1, 2, 0, 4], 'col_3': [1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], "target": [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0]}
    df_test = pd.DataFrame.from_dict(data)
    enc = FoldCounters(n_folds=3)
    enc.fit(df_test[['col_1', 'col_2', 'col_3']], df_test['target'], seed=1)
    counts = enc.transform(df_test[['col_1', 'col_2', 'col_3']], a=1, b=2)
    ans = np.array([[0.3333333333333333, 0.3, 0.5797101449275363, 0.2857142857142857, 0.35, 0.547112462006079, 0.5384615384615384, 0.65, 0.5805515239477503 ],\
[0.2857142857142857, 0.35, 0.547112462006079, 0.42857142857142855, 0.35, 0.60790273556231, 0.6666666666666666, 0.6, 0.641025641025641 ],\
[0.2857142857142857, 0.35, 0.547112462006079, 0.42857142857142855, 0.35, 0.60790273556231, 0.5, 0.4, 0.625 ],\
[0.7142857142857143, 0.35, 0.729483282674772, 0.5, 0.3, 0.6521739130434783, 0.6923076923076923, 0.65, 0.6386066763425254 ],\
[0.7142857142857143, 0.35, 0.729483282674772, 1.0, 0.2, 0.9090909090909091, 0.6666666666666666, 0.6, 0.641025641025641 ],\
[0.3333333333333333, 0.45, 0.5442176870748299, 0.3333333333333333, 0.15, 0.6201550387596899, 0.2857142857142857, 0.35, 0.547112462006079 ],\
[0.2857142857142857, 0.35, 0.547112462006079, 0.42857142857142855, 0.35, 0.60790273556231, 0.6666666666666666, 0.6, 0.641025641025641 ],\
[0.7142857142857143, 0.35, 0.729483282674772, 1.0, 0.2, 0.9090909090909091, 0.5, 0.4, 0.625 ],\
[0.3333333333333333, 0.45, 0.5442176870748299, 0.25, 0.2, 0.5681818181818181, 0.2857142857142857, 0.35, 0.547112462006079 ],\
[0.3333333333333333, 0.3, 0.5797101449275363, 1.0, 0.15, 0.9302325581395349, 0.5384615384615384, 0.65, 0.5805515239477503 ],\
[0.375, 0.4, 0.5729166666666667, 1.0, 0.05, 0.9756097560975611, 0.6923076923076923, 0.65, 0.6386066763425254 ],\
[0.8, 0.25, 0.8, 0.2857142857142857, 0.35, 0.547112462006079, 0.5384615384615384, 0.65, 0.5805515239477503 ],\
[0.3333333333333333, 0.3, 0.5797101449275363, 0.25, 0.2, 0.5681818181818181, 0.2857142857142857, 0.35, 0.547112462006079 ],\
[0.8333333333333334, 0.3, 0.7971014492753624, 1.0, 0.15, 0.9302325581395349, 0.6666666666666666, 0.6, 0.641025641025641 ],\
[0.375, 0.4, 0.5729166666666667, 0.5, 0.3, 0.6521739130434783, 0.6923076923076923, 0.65, 0.6386066763425254 ],\
[0.8, 0.25, 0.8, 0.2857142857142857, 0.35, 0.547112462006079, 0.2857142857142857, 0.35, 0.547112462006079 ],\
[0.3333333333333333, 0.3, 0.5797101449275363, 1.0, 0.15, 0.9302325581395349, 0.5384615384615384, 0.65, 0.5805515239477503 ],\
[0.6, 0.25, 0.7111111111111111, 0.0, 0.15, 0.46511627906976744, 0.6923076923076923, 0.65, 0.6386066763425254 ],\
[0.2857142857142857, 0.35, 0.547112462006079, 1.0, 0.2, 0.9090909090909091, 0.6666666666666666, 0.6, 0.641025641025641 ],\
[0.375, 0.4, 0.5729166666666667, 0.5, 0.3, 0.6521739130434783, 0.6923076923076923, 0.65, 0.6386066763425254 ],\
[0.6, 0.25, 0.7111111111111111, 1.0, 0.2, 0.9090909090909091, 0.2857142857142857, 0.35, 0.547112462006079 ],\
[0.375, 0.4, 0.5729166666666667, 0.0, 0.15, 0.46511627906976744, 0.2857142857142857, 0.35, 0.547112462006079 ],\
[0.7142857142857143, 0.35, 0.729483282674772, 1.0, 0.05, 0.9756097560975611, 0.2857142857142857, 0.35, 0.547112462006079 ],\
[0.8333333333333334, 0.3, 0.7971014492753624, 0.25, 0.2, 0.5681818181818181, 0.5, 0.4, 0.625 ],\
[0.6, 0.25, 0.7111111111111111, 0.5, 0.3, 0.6521739130434783, 0.2857142857142857, 0.35, 0.547112462006079 ],\
[0.2857142857142857, 0.35, 0.547112462006079, 1.0, 0.15, 0.9302325581395349, 0.6666666666666666, 0.6, 0.641025641025641 ],\
[0.7142857142857143, 0.35, 0.729483282674772, 0.5, 0.3, 0.6521739130434783, 0.6923076923076923, 0.65, 0.6386066763425254 ],\
[0.7142857142857143, 0.35, 0.729483282674772, 0.5, 0.1, 0.7142857142857143, 0.6666666666666666, 0.6, 0.641025641025641 ],\
[0.8, 0.25, 0.8, 0.25, 0.2, 0.5681818181818181, 0.5384615384615384, 0.65, 0.5805515239477503 ],\
[0.3333333333333333, 0.45, 0.5442176870748299, 0.3333333333333333, 0.15, 0.6201550387596899, 0.5384615384615384, 0.65, 0.5805515239477503 ]])
    assert len(counts.shape) == 2
    assert counts.shape[0] == 30
    assert counts.shape[1] == 9
    assert np.allclose(counts, ans, atol=1e-8)
    assert type(counts) == np.ndarray

test_fold_counters_big()