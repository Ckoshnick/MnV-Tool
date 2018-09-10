# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 12:34:37 2018

@author: koshnick
"""

import matplotlib.pyplot as plt
import mnv
import pandas as pd
import cProfile
from PI_client import pi_client
import time

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 500)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('{}  {} ms'.format(method.__name__,
                  round((te - ts) * 1000), 2))
        return result

    return timed


@timeit
def dk_test_init(data):
    '''
    basic functionality test

         IQRmult = 3,
         IQR = 'y',
         floor = 0,
         ceiling = 10000,
         resampleRate = 'D',
         sliceType = 'half',
         midDate = None,
         dateRanges = None,
         OATsource = 'file',
         OATname = None
    '''
    print('dk_test_init')
    inputDict = {'IQRmult': 3.0, 'floor': 0, 'ceiling': 10000, 'IQR': True,
                 'resampleRate': 'H',
                 'sliceType': 'half',  # half, middate, ranges
                 'midDate': '2018-03-01',
                 'dateRanges': ['2016-01-01', '2017-01-01', '2017-01-02', '2018-01-01'],
                 'OATsource': 'file', 'OATname': None}

    dk = mnv.data_keeper(data[data.columns[0]].to_frame(), inputDict)
    print('dk_test_init passed')

    return dk


@timeit
def dk_test_cleaning(data):
    '''
    Turn this into a function to test
    Test the manual cleaning suite by calling all cleaning functions here

    These functions should be the same as dk.default_clean()

    check pre post to verify all worked right
    '''

    dk = dk_test_init(data)

    # Default Clean functions
    dk.remove_outliers()
    dk._resample(resampleRate='H', aggFun='mean')
    dk.add_degree_hours(cutoff=65)
    dk._resample(aggFun='sum')
    dk.add_time_columns()
    dk.data_slice()

    return dk


@timeit
def test_oat_self(data):

    print('needs fixing!!!!!!!!!!!')
    # TODO: Fix

    print('dk_test_oat_self')
    inputDict = {'IQRmult': 3.0, 'floor': 0, 'ceiling': 10000, 'IQR': True,
                 'resampleRate': 'D',
                 'sliceType': 'midDate',  # half, middate, ranges
                 'midDate': '2018-03-01',
                 'dateRanges': ['2016-01-01', '2017-01-01', '2017-01-02', '2018-01-01'],
                 'OATsource': 'self',
                 'OATname': 'OAT'}

    dk = mnv.data_keeper(data[data.columns[0:2]], inputDict)
    dk.default_clean()

    dk.modifiedData['HDH'].plot()
    dk.modifiedData['CDH'].plot()

    return dk


@timeit
def test_jupyter(dk, mod):

    dk._resampled_plot()
    dk._pre_post_plot()

    allmod = mnv.many_ols(dk.pre, dk.post, mod.params.__dict__)

    allmod.run_all_linear()
    print(allmod.statsPool[0:5])
    allmod.plot_pool(0)

    mod.model_plot()
    mod.calculate_vif()
    mod.calculate_kfold()
    mod.calculate_F_uncertainty()

    mod.savings_plot()

    mod.generate_savings_summary()
    mod.plot_tmy_comparison()


@timeit
def test_kfold(dk, mod):

    print(mod._folds)
    mod.calculate_kfold()
    print(mod.kfoldStats)

    return mod


@timeit
def test_archive(dk, mod):

    mod.calculate_vif()
    mod.calculate_kfold()

    mod.model_plot()
    mod.savings_plot()

    mod.calculate_uncertainty()
    mod.calculate_tmy_models()
    mod.compile_savings()

    mnv.create_archive(dk, mod, saveFigs=True)

    return dk, mod


def test():
    filePath = 'data/arc2yeardata.xlsx'
    data = pd.read_excel(filePath, header=0, index_col=0, parse_dates=True)
    dk, mod = generic_loader(data)
    return dk, mod


@timeit
def test_tmy(dk, mod):
#    dk._outlier_plot()
#    dk._resampled_plot()
#    print(mod._generate_savings_intervals())
    mod.generate_savings_summary()

    return dk, mod


@timeit
def generic_loader(data):
    inputDict = {'IQRmult': 3.0,
                 'IQR': True,
                 'resampleRate': 'D',  # 'D' for daily 'H' for hourly
                 'sliceType': 'ranges',  # half, middate, ranges
                 'midDate': '2017-01-01',
                 'dateRanges': ['2017-01-01', '2018-01-01', '2018-01-11', '2018-06-20'],
                 'OATsource': 'file',  # 'self' or 'file'
                 'OATname': 'OAT',  #
                 }

    dk = mnv.data_keeper(data, inputDict)
    dk.default_clean()

    modelDict = {'var': 'CDH + HDH + C(month)',
                 'testTrainSplit': 'random',
                 'randomState': None,
                 'testSize': 0.2,
                 'commodityRate': 0.056,
                 'varPermuteList': ['', 'C(weekday)', 'C(month)']}

    mod = mnv.ols_model(dk.pre, dk.post, modelDict)

    return dk, mod


if __name__ == "__main__":
    filePath = '../data/pes kbtu.xlsx'
    data = pd.read_excel(filePath, header=0, index_col=0, parse_dates=True)
    dk, mod = generic_loader(data)

#    mod = test_kfold(dk, mod)
#    dk, mod = test_archive(dk, mod)
#    dk, mod = test_tmy(dk, mod)
    test_jupyter(dk, mod)
    pass


