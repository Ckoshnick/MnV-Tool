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

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 500)

def dc_test_init(data):
    '''
    basic functionality test

         IQRmult = 3,
         IQR = 'y',
         floor = 0,
         ceiling = 10000,
         resampleRate = 'D',
         verbosity = 2,
         sliceType = 'half',
         midDate = None,
         dateRanges = None,
         OATsource = 'file',
         OATname = None
    '''
    print('dc_test_init')
    inputDict = {'IQRmult' : 3.0, 'floor' : 0, 'ceiling': 10000, 'IQR' : 'y',
             'resampleRate' : 'H',
             'verbosity' : 3,
             'sliceType' : 'half', #half, middate, ranges
             'midDate' : '2018-03-01', #only needed with sliceType : 'middate'
             'dateRanges' : ['2016-01-01','2017-01-01','2017-01-02','2018-01-01'],
             'OATsource' : 'file', 'OATname' : None}

    dc = mnv.data_handler(data[data.columns[0]].to_frame(), inputDict)
    print('dc_test_init passed')

    return dc


def dc_test_cleaning(data):
    '''
    Turn this into a function to test
    Test the manual cleaning suite by calling all cleaning functions here

    These functions should be the same as dc.default_clean()

    check pre post to verify all worked right
    '''

    dc = dc_test_init(data)

    # Default Clean functions
    dc.remove_outliers()
    dc._resample(resampleRate = 'H', aggFun = 'mean')
    dc.add_degree_hours(cutoff=65)
    dc._resample(aggFun = 'sum')
    dc.add_time_columns()
    dc.data_slice()

    return dc

def mc_test_single(dc):
    """
    """
    ## INPUT PARAMS
    modelDict = {'params': 'CDH + HDH + C(month) + C(hour)',
                 'testTrainSplit': 'random',
                 'randomState': 42990,
                 'testSize': 0.2,
                 'commodityRate': 0.056,
                 'paramPermuteList': ['','C(month)','C(weekday)']}
    
    mod1 = mnv.ols_model(dc.pre, dc.post, modelDict)
#    mod1.model_plot2()
#    mod1.stats_plot()
#    mod1.savings_plot()
    
    return mod1, dc

def mc_test_all_linear(dc):
    """
    """
    ## INPUT PARAMS    
    modelDict = {'params': 'CDH + HDH + C(month) + C(weekday)',
                 'testTrainSplit': 'random',
                 'randomState': 4291990,
                 'testSize': 0.2,
                 'commodityRate': 0.056,
                 'paramPermuteList': ['','C(month)','C(weekday)']}
    
    mc = mnv.many_ols(dc.pre, dc.post, modelDict)
    mc.run_all_linear()

    return mc, dc

def test_VIF(mc):
    
    mc.calculate_vif()
    
def test_oat_self(data):

    print('dc_test_oat_self')
    inputDict = {'IQRmult' : 3.0, 'floor' : 0, 'ceiling': 10000, 'IQR' : 'y',
             'resampleRate' : 'D',
             'verbosity' : 3,
             'sliceType' : 'midDate', #half, middate, ranges
             'midDate' : '2018-03-01', #only needed with sliceType : 'middate'
             'dateRanges' : ['2016-01-01','2017-01-01','2017-01-02','2018-01-01'],
             'OATsource' : 'file',
             'OATname' : 'OAT'}

    dc = mnv.data_handler(data[data.columns[0:2]], inputDict)
    dc.default_clean()
    
    dc.modifiedData['HDH'].plot()
    dc.modifiedData['CDH'].plot()

    return dc

    
def test_jupyter(data):

    
    inputDict = {'IQRmult' : 4.0,
             'IQR' : 'y', 
             'resampleRate' : 'H', #'D' for daily 'H' for hourly
             'verbosity' : 3,
             'sliceType' : 'ranges', #half, middate, ranges
             'midDate' : '2017-01-01', #only needed with sliceType : 'middate'
             'dateRanges' : ['2017-01-01','2018-01-01','2018-01-02','2018-04-01'], #only needed with sliceType : 'ranges'
             'OATsource' : 'file', #'self' or 'file'
             'OATname' : 'OAT', #Name of OAT column if OATsource is 'self'} #only needed with sliceType : 'ranges'
            }
    
    dc = mnv.data_handler(data[data.columns[1]], inputDict)
    dc.default_clean()
    
    
    modelDict = {'params': 'CDH + HDH + C(daytime) +C(month)',
             'testTrainSplit': 'random',
             'randomState': None,
             'testSize': 0.5,
             'commodityRate': 0.056,
             'paramPermuteList': ['', 'C(daytime)', 'C(weekday)', 'C(month)']}

    allmod = mnv.many_ols(dc.pre, dc.post, modelDict)
    
    allmod.run_all_linear()
    print(allmod.statsPool[0:5])
    allmod.plot_pool(0)
    modelDict['params'] = allmod.statsPool.iloc[0]['params']
    
    mod = mnv.ols_model(dc.pre, dc.post, modelDict)
    mod.model_plot()
    mod.calculate_vif()
    
    plt.show() # Show plot before Stats summary
    
    print(mod.vif)
    mod.Fit.summary()
    
def test_new_kfold(data):
    inputDict = {'IQRmult' : 4.0,
         'IQR' : 'y', 
         'resampleRate' : 'H', #'D' for daily 'H' for hourly
         'verbosity' : 3,
         'sliceType' : 'ranges', #half, middate, ranges
         'midDate' : '2017-01-01', #only needed with sliceType : 'middate'
         'dateRanges' : ['2017-01-01','2018-01-01','2018-01-02','2018-04-01'], #only needed with sliceType : 'ranges'
         'OATsource' : 'file', #'self' or 'file'
         'OATname' : 'OAT', #Name of OAT column if OATsource is 'self'} #only needed with sliceType : 'ranges'
        }
    
    dc = mnv.data_handler(data[data.columns[1]], inputDict)
    dc.default_clean()
    
    modelDict = {'params': 'CDH + HDH + C(daytime) +C(month)',
             'testTrainSplit': 'random',
             'randomState': None,
             'testSize': 0.2,
             'commodityRate': 0.056,
             'paramPermuteList': ['', 'C(daytime)', 'C(weekday)', 'C(month)']}
    
    mod = mnv.ols_model(dc.pre, dc.post, modelDict)
    
    return mod

    

if __name__ == "__main__":
    #    filePath = 'data/OATtest.xlsx'
    filePath = 'data/zone temps.xlsx'
#    filePath = 'data/arc2yeardata.xlsx'
    data = pd.read_excel(filePath, header=0, index_col=0, parse_dates=True)
    
#    test_jupyter(data)
    mk = test_new_kfold(data)

        
    

    
