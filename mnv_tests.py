# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 12:34:37 2018

@author: koshnick
"""

import matplotlib.pyplot as plt
import mnv14 as mnv
import pandas as pd
import cProfile
import pstats
from PI_client import pi_client

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 500)

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
    inputDict = {'IQRmult' : 3.0, 'floor' : 0, 'ceiling': 10000, 'IQR' : 'y',
             'resampleRate' : 'H',
             'sliceType' : 'half', #half, middate, ranges
             'midDate' : '2018-03-01', #only needed with sliceType : 'middate'
             'dateRanges' : ['2016-01-01','2017-01-01','2017-01-02','2018-01-01'],
             'OATsource' : 'file', 'OATname' : None}

    dk = mnv.data_keeper(data[data.columns[0]].to_frame(), inputDict)
    print('dk_test_init passed')

    return dk


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
    dk._resample(resampleRate = 'H', aggFun = 'mean')
    dk.add_degree_hours(cutoff=65)
    dk._resample(aggFun = 'sum')
    dk.add_time_columns()
    dk.data_slice()

    return dk

def mc_test_single(dk):
    """
    """
    ## INPUT PARAMS
    modelDict = {'params': 'CDH + HDH + C(month) + C(hour)',
                 'testTrainSplit': 'random',
                 'randomState': 42990,
                 'testSize': 0.2,
                 'commodityRate': 0.056,
                 'paramPermuteList': ['','C(month)','C(weekday)']}

    mod1 = mnv.ols_model(dk.pre, dk.post, modelDict)
#    mod1.model_plot2()
#    mod1.stats_plot()
#    mod1.savings_plot()

    return mod1, dk

def mc_test_all_linear(dk):
    """
    """
    ## INPUT PARAMS
    modelDict = {'params': 'CDH + HDH + C(month) + C(weekday)',
                 'testTrainSplit': 'random',
                 'randomState': 4291990,
                 'testSize': 0.2,
                 'commodityRate': 0.056,
                 'paramPermuteList': ['','C(month)','C(weekday)']}

    mc = mnv.many_ols(dk.pre, dk.post, modelDict)
    mc.run_all_linear()

    return mc, dk

def test_VIF(mc):

    mc.calculate_vif()

def test_oat_self(data):

    print('dk_test_oat_self')
    inputDict = {'IQRmult' : 3.0, 'floor' : 0, 'ceiling': 10000, 'IQR' : 'y',
             'resampleRate' : 'D',
             'sliceType' : 'midDate', #half, middate, ranges
             'midDate' : '2018-03-01', #only needed with sliceType : 'middate'
             'dateRanges' : ['2016-01-01','2017-01-01','2017-01-02','2018-01-01'],
             'OATsource' : 'file',
             'OATname' : 'OAT'}

    dk = mnv.data_keeper(data[data.columns[0:2]], inputDict)
    dk.default_clean()

    dk.modifiedData['HDH'].plot()
    dk.modifiedData['CDH'].plot()

    return dk


def test_jupyter(data):


    inputDict = {'IQRmult' : 4.0,
             'IQR' : 'y',
             'resampleRate' : 'H', #'D' for daily 'H' for hourly
             'sliceType' : 'ranges', #half, middate, ranges
             'midDate' : '2017-01-01', #only needed with sliceType : 'middate'
             'dateRanges' : ['2016-01-01','2017-01-01','2017-01-02','2018-04-01'], #only needed with sliceType : 'ranges'
             'OATsource' : 'file', #'self' or 'file'
             'OATname' : 'OAT', #Name of OAT column if OATsource is 'self'} #only needed with sliceType : 'ranges'
            }

    dk = mnv.data_keeper(data, inputDict)
    dk.default_clean()


    modelDict = {'params': 'CDH + HDH + C(daytime) +C(month)',
             'testTrainSplit': 'random',
             'randomState': None,
             'testSize': 0.5,
             'commodityRate': 0.056,
             'paramPermuteList': ['', 'C(daytime)', 'C(weekday)', 'C(month)']}

    allmod = mnv.many_ols(dk.pre, dk.post, modelDict)

    allmod.run_all_linear()
    print(allmod.statsPool[0:5])
    allmod.plot_pool(0)
    modelDict['params'] = allmod.statsPool.iloc[0]['params']

    mod = mnv.ols_model(dk.pre, dk.post, modelDict)
    mod.model_plot()
    mod.calculate_vif()

    plt.show() # Show plot before Stats summary

    print(mod.vif)
    mod.Fit.summary()

def test_new_kfold(data):
    inputDict = {'IQRmult' : 3.0,
         'IQR' : 'y',
         'resampleRate' : 'H', #'D' for daily 'H' for hourly
         'sliceType' : 'ranges', #half, middate, ranges
         'midDate' : '2017-01-01', #only needed with sliceType : 'middate'
         'dateRanges' : ['2017-01-01','2018-01-01','2018-01-02','2018-04-01'], #only needed with sliceType : 'ranges'
         'OATsource' : 'file', #'self' or 'file'
         'OATname' : 'OAT', #Name of OAT column if OATsource is 'self'} #only needed with sliceType : 'ranges'
        }

    dk = mnv.data_keeper(data, inputDict)
    dk.default_clean()

    modelDict = {'params': 'CDH + HDH + C(daytime) +C(month)',
             'testTrainSplit': 'random',
             'randomState': None,
             'testSize': 0.2,
             'commodityRate': 0.056,
             'paramPermuteList': ['', 'C(daytime)', 'C(weekday)', 'C(month)']}

    mod = mnv.ols_model(dk.pre, dk.post, modelDict)

    return mod

def generic_loader(data):
    inputDict = {'IQRmult' : 3.0,
         'IQR' : 'y',
         'resampleRate' : 'D', #'D' for daily 'H' for hourly
         'sliceType' : 'ranges', #half, middate, ranges
         'midDate' : '2017-01-01', #only needed with sliceType : 'middate'
         'dateRanges' : ['2017-01-01','2018-01-01','2018-01-16','2018-04-01'], #only needed with sliceType : 'ranges'
         'OATsource' : 'file', #'self' or 'file'
         'OATname' : 'OAT', #Name of OAT column if OATsource is 'self'} #only needed with sliceType : 'ranges'
        }

    dk = mnv.data_keeper(data, inputDict)
    dk.default_clean()

    modelDict = {'params': 'CDH + HDH + C(month)',
             'testTrainSplit': 'random',
             'randomState': None,
             'testSize': 0.2,
             'commodityRate': 0.056,
             'paramPermuteList': ['', 'C(daytime)', 'C(weekday)', 'C(month)']}


    mod = mnv.ols_model(dk.pre, dk.post, modelDict)

    return dk, mod

def test_archive():
    filePath = 'data/arc2yeardata.xlsx'
    data = pd.read_excel(filePath, header=0, index_col=0, parse_dates=True)

    dk, mod = generic_loader(data)

    mod.calculate_kfold()

    mod.model_plot()
    mod.savings_plot()

    mnv.create_archive(dk, mod, saveFigs=True)

    return dk, mod

def test():
    filePath = 'data/arc2yeardata.xlsx'
    data = pd.read_excel(filePath, header=0, index_col=0, parse_dates=True)
    dk, mod = generic_loader(data)
    return dk, mod

def test_tmy():
#    filePath = 'data/pes kbtu.xlsx'
#    data = pd.read_excel(filePath, header=0, index_col=0, parse_dates=True)

#    data.columns = ['chw_chw','ele/ele','stm/stm']

    pi = pi_client()
    tag = pi.search_by_point('*pes*chil*kbtu')[0]
    data  = pi.get_stream_by_point(tag, start='2017-01-01', end='y')
    dk, mod = generic_loader(data)

    dk._outlier_plot()
    dk._resampled_plot()

    mod.compare_tmy_models()

    return dk, mod




if __name__ == "__main__":
#    filePath = 'data/OATtest.xlsx'
#    filePath = 'data/arc2yeardata.xlsx'
#    filePath = 'data/arc2yeardata.xlsx'
#    data = pd.read_excel(filePath, header=0, index_col=0, parse_dates=True)

#    test_jupyter(data)
#    mk = test_new_kfold(data)
#    dk, mod = test_archive()
    dk, mod = test_tmy()
    pass






