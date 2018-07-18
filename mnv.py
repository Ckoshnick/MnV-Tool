# -*- coding: utf-8 -*-
"""
Created on Wed Apr 04 10:07:41 2018

@author: koshnick

Last update 5/14/18 - koshnick
Added Params subclass to data container. Refactored code to fit new convention
Next time - need to add internal functions to _convention.

"""

import sys
import math
import mypy
import itertools

import numpy as np
import pandas as pd

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from os import path
from datetime import datetime
from sklearn import linear_model
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score

#path_prefix = path.dirname(path.abspath(__file__))

sys.path.append('../../mypy')
plt.rcParams.update({'figure.max_open_warning': 0})

figW = 18
figH = 6

class data_handler():
    """
    input **kwargs
    inputDict = {'IQRmult' : 3.0,
                 'IQR' : 'y',
                 'floor': 0,
                 'ceiling': 10000,

                 'resampleRate' : 'D',
                 'verbosity' : 2,
                 'sliceType' : 'half', #half, middate, ranges
                 'midDate' : None, #only needed with sliceType : 'middate'
                 'dateRanges' : ['20--','20--','20--','20--'], #required for 'ranges'
                 'OATname' : None,
                 'OATsource' : 'file'}
    """


    class parameters():
        # Params Init
        def __init__(self,
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
                 OATname = None):

            self.OATsource = OATsource
            self.floor = floor
            self.ceiling = ceiling
            self.IQR = IQR
            self.IQRmult = IQRmult
            self.resampleRate = resampleRate
            self.sliceType = sliceType
            self.midDate = midDate
            self.dateRanges = dateRanges
            self.verbosity = verbosity #Verbosity of outputs (plots and stats, 0-4)
            self.OATname = OATname

        def show(self):
            for k,v in self.__dict__.iteritems():
                print(k,v)


    # Data Class init
    def __init__(self, data, inputDict=None):

        self.params = self.parameters(**inputDict)

        if isinstance(data,pd.DataFrame):
            self.rawData = data[data.columns[0]].to_frame()
            print('Loaded pd.DataFrame - using column "{}"'.format(data.columns[0]))
        elif isinstance(data,pd.Series):
            self.rawData = data.to_frame()

        if self.params.OATsource == 'self':
            self.OAT = data[self.params.OATname]

        self.modifiedData = self.rawData.copy()
        self.com = self.modifiedData.columns[0]

        self.date = datetime.now()
        self.name = '{} {}'.format(self.com, str(self.date))


    def commandLine(self):
        """
        This function should serve as a method to help guide the user
        through all of the necessary processing steps to create a model.
        The strategy is to fill in variables that are missing
        Run functions and ask user if output seems appropriate
        Re-run functions if needed
        General model(s)
        Plot output of model(s)
        Display stats of model(s)
        """
        pass


    def undo(self):
        '''
        Allows the user to go back one step in data modification
        Should this function plot? on verbosity?
        '''
        self.modifiedData = self.restoreData.copy()


    def interval_checker():
        """
        Calculate the intervalsize of a time series dataset and returns
        'hourly', '15 minute', 'daily', 'monthly'
        """
        pass

#==============================================================================
# CLEANING
#==============================================================================

    def _IQR(self):
        """
        Calculates new boundaries based on the Inner Quartile Range multiplied
        by a IQR multiplier (default 1.5) ##CHECK
        IQRmult taken from object attribute

        Passes new upper and lower bounds to the remove_outliers. Boundarires
        change only if they are more restrictive than the Floor and Cieling
        limits provided to remove_outliers
        """

        Q1 = self.modifiedData[self.com].quantile(0.25)
        Q3 = self.modifiedData[self.com].quantile(0.75)
        IQR = Q3 - Q1

        print('Q(75%): {0:.2f} Q(25%): {1:.2f}'.format(Q3, Q1))
        print('IQR value is {0:.2f}'.format(IQR))

        upper = Q3 + (self.params.IQRmult * IQR)
        lower = Q1 - (self.params.IQRmult * IQR)

        return upper, lower


    def remove_outliers(self,
                        floor=None,
                        ceiling=None,
                        IQR=None):
        """
        Tests if data is between bounds (floor, ceiling) and removes entries
        that are out of bounds

        inputs:
            floor - int - default = 0:
                The lowest number that the data should take on
            ceiling - int - default = ??: ##CHECK
                The largest number that the data should take on
            iqr - bool:
                If iqr true, then adjust the floor and ceiling with method _IQR

        output:
            modifies self.modifiedData inplace
            return self

        """
        self.restoreData = self.modifiedData.copy()
        temp = self.modifiedData.copy()

        if floor == None: floor = self.params.floor
        if ceiling == None: ceiling = self.params.ceiling
        if IQR == None: IQR = self.params.IQR

        if IQR:
            IQRupper, IQRlower = self._IQR()

            if self.params.verbosity > 4:
                print('IQRupper', IQRupper, 'IQRlower', IQRlower)

            if IQRupper < ceiling:
                ceiling = IQRupper
                print('Ceiling adjusted by IQR - Now {0:.2f}'.format(ceiling))
            if IQRlower > floor:
                floor = IQRlower
                print('Floor adjusted by IQR   - Now {0:.2f}'.format(floor))

        print(floor)

        if floor != None:
            temp = temp.where(temp > floor, other=np.nan)

        if ceiling != None:
            temp = temp.where(temp < ceiling, other=np.nan)
        else:
            pass

        temp.dropna(inplace=True)

        if self.params.verbosity > 2:
            self._outlier_plot(temp)

        self.modifiedData[self.com] = temp
        return self ## CHECK do i need this here?


    def add_oat_column(self):
        """
        Call mypy to add OAT data to the commodity data. Only need to call this
        if the input data was not supplied with assocaited OAT data.
        """

        self.restoreData = self.modifiedData.copy()

        if self.OATsource == 'file':
            self.modifiedData = mypy.merge_oat(self.modifiedData, choose='y') ##CHECK rework MYPY and this function

        elif self.OATsource == 'self':
            self.modifiedData['OAT'] = self.OAT

        print('Created OAT from {}'.format(self.OATsource))
        print('')

        return self


    def _resample(self, resampleRate=None, aggFun = 'mean'):

        self.restoreData = self.modifiedData.copy()

        if resampleRate == None:
            resampleRate = self.params.resampleRate

        self.modifiedData = self.modifiedData.resample(resampleRate).agg(aggFun)
        self.params.modifiedDataInterval = resampleRate


    def add_degree_hours(self, cutoff=65):
        """
        Use OAT data to generate CDD and HDD. In this implementation we sample
        the HD-hours and CD-hours and then sum them up to what ever timestep
        we're using for the M&V (usually daily) ##CHECK make sure this works
        if we want to make an hourly model

        """
        self.restoreData = self.modifiedData.copy()
        ## Check redundant code - refactor


        # TODO: Should not use resample rate here, should use actual invterval
        # assert that it is H or D??
        if self.params.OATsource == 'file':
            hours = mypy.calculate_degree_days(by=self.params.resampleRate,
                                               cutoff = 65)
        elif self.params.OATsource == 'self':
            hours = mypy.calculate_degree_days(data=self.modifiedData,
                                               by=self.params.resampleRate,
                                               cutoff = 65)

        self.modifiedData['HDH'] = hours['HDH']
        self.modifiedData['CDH'] = hours['CDH']

        self.modifiedData['HDH2'] = self.modifiedData['HDH'] ** 2 ## Check. How do we sum HDH2 vs HDD2?
        self.modifiedData['CDH2'] = self.modifiedData['CDH'] ** 2

        return self

    def add_time_columns(self):
        self.modifiedData = mypy.build_time_columns(self.modifiedData)


    def data_slice(self):
        '''
        If pre and post periods are loaded into the class as a single object
        this function will allow it to be sliced into how=
         -- half: 1st half 2nd half
         -- middate: a single central date to be the split point
         -- ranges: specifying the two date ranges

        example datestring : "2017-06-01" for June 01, 2017

        returns: None
        Modifies instance inplace

        '''

        how = self.params.sliceType

        if how == 'half':
            dataLength = len(self.modifiedData)
            midPoint = int(math.floor(dataLength)/2)

            self.pre = self.modifiedData.iloc[0:midPoint,:]
            self.post = self.modifiedData.iloc[midPoint:,:]

        elif how == 'middate':
            if self.params.midDate:
                midDate = pd.to_datetime(self.params.midDate)

                #iloc to avoid duplicating the point at middate
                self.pre = self.modifiedData[:midDate].iloc[:-1]
                self.post = self.modifiedData[midDate:]
            else:
                raise ValueError('data_slice: Must supply midDate with how = "middate"')

        elif how == 'ranges':
            if self.params.dateRanges:
                ranges = self.params.dateRanges ## CHECK lazy spaghetti
                preStart = pd.to_datetime(ranges[0])
                preEnd = pd.to_datetime(ranges[1])
                postStart = pd.to_datetime(ranges[2])
                postEnd = pd.to_datetime(ranges[3])

                self.pre = self.modifiedData[preStart:preEnd]
                self.post = self.modifiedData[postStart:postEnd]
            else:
                raise ValueError('data_slice: Must supply datranges for how = "ranges"')

    def default_clean(self):

        """
        A suite of outlier cleaning functions
        ##CHECK add more detail

        """

        self.remove_outliers()
        self._resample(resampleRate = 'H', aggFun = 'mean')
        self.add_degree_hours()
        self._resample(aggFun = 'sum')
        self.add_time_columns()
        self.data_slice()


#==============================================================================
# PLOTTING
#==============================================================================

    def _outlier_plot(self, temp):

        fig = plt.figure(figsize=(figW, figH))
        ax0 = plt.subplot2grid((1,5), (0,0))

        ## Box plot
        ax1 = plt.subplot2grid((1, 5), (0, 0), colspan=1)

        _comb = pd.concat([self.modifiedData[self.com],temp], axis=1)
        _comb.columns = ['before','after']
        sns.boxplot(data=_comb, ax=ax1)
        plt.ylabel(self.com)
        plt.title('Remove Outlier Boxplot')


        ## Scatter plot
        ax2 = plt.subplot2grid((1, 5), (0, 1), colspan=4)

        indexDifference = self.modifiedData.index.difference(temp.index)

        plt.plot(temp, color= 'k', linestyle='', marker='.')
        plt.plot(self.modifiedData[self.com][indexDifference], color = 'r',
                 linestyle='', marker='.')

        plt.title('Outliers removed')
        plt.show()

        """
        might need a copy of this method after i edited it ## Check
        def model_plot(self, pre1,pre2, post1,post2):

        fig = plt.figure(figsize=(figW, figH*1.5))
        ax0 = plt.subplot2grid((2,1), (0,0))

        ax1 = plt.subplot2grid((2, 1), (0, 0), colspan=1)

        self.pre[self.com].plot(label='actual')
        self.preModel.plot(label='model')
        plt.title('preModel ' + params)
        plt.ylabel(self.com)
        plt.legend()

        ax2 = plt.subplot2grid((2, 1), (1, 0), colspan=1)

        self.post[self.com].plot(label='actual')
        self.postModel.plot(label='model')
        plt.ylabel(self.com)
        plt.title('postModel ' + params)
        plt.legend()


        """

#==============================================================================
# MODELING
#==============================================================================

class many_ols():
    """
    Similar to the data holding class, will allow the user to put their data into
    an object that will cleanly allow them to run different models and view
    the stats and plots


    """
    def __init__(self, pre, post, folding='simple', inputParams=None):
        self.pre = pre
        self.post = post
        self.inventory = None
        self.folding = folding
        self.foldParameter = 0.7
        self.com = self.pre.columns[0]
        self.inputParams = inputParams

        if self.folding == 'simple':

            length = len(self.pre)
            sliceIndex = int(round(length * self.foldParameter))

            self.train = self.pre.iloc[:sliceIndex]
            self.test = self.pre.iloc[sliceIndex:length]

    def _param_permute(self):

        """
        Take in params and return a list of all combos of params without making
        stupid combos
        """

        a1 = ['CDH', 'CDH2', '']
        b1 = ['HDH', 'HDH2', '']

        if self.inputParams == None:
            inputs = ['', 'C(month)','C(hour)','C(weekend)']
        else:
            inputs = self.inputParams

        els = []
        for i in range(1,len(inputs)):
            els += [list(x) for x in itertools.combinations(inputs, i)]

        parList = []

        for a in a1:
            for b in b1:
                for e in els:
                    par = [a,b] + e
                    parList.append(" + ".join(filter(None, par)).rstrip(' +'))
        parList.remove('')

        return parList

    def run_all_linear(self):
        """
        Run several models using the prebaked vars in _param_permute. ## check
        need to refactor the vars as things that can be modified from the outside
        Collect all of the models in modelPool to have stats run on them later
        """

        outputs = {}
        paramList = self._param_permute()

        for params in paramList:
            try:
                outputs[params] = ols_model(self.pre, self.post, params)
            except Exception as e:
                print('Could not complete model with {}'.format(params))
                print(e)

        self._modelPool = outputs
        self._pool_stats()


    def _pool_stats(self):

        """
        Take all of the models in modelPool and make sense of them so they
        can be ranked, plotted, etc..

        """

        try:
            assert(self._modelPool)
        except AssertionError:
            print('You must run "run_all_linear" to generate modelPool before running pool_stats')

        statsPool = {}

        modelNumber = -1

        for params, mod in self._modelPool.iteritems():
            modelNumber += 1

            newStatsRow = {}

            newStatsRow['params'] = params
            newStatsRow['AIC'] = mod.Fit.aic
            newStatsRow['BIC'] = mod.Fit.bic
            newStatsRow['R2'] = mod.Fit.rsquared
            newStatsRow['AR2'] = mod.Fit.rsquared_adj
            newStatsRow['mse'] = mod.Fit.mse_resid
            newStatsRow['sum'] = mod.Fit.summary()

            statsPool[modelNumber] = newStatsRow

        self.statsPool = pd.DataFrame(statsPool).T.sort_values('AIC')


    def plot_pool(self, number=5):

        for i in range(number):
            modParams = self.statsPool['params'].iloc[i]
            tempMod = self._modelPool[modParams]

            tempMod.model_plot()
            tempMod.stats_plot()
            plt.show()


class ols_model():

    def __init__(self, pre, post, params, rate = 1):

        self.params = params
        self.pre = pre
        self.post = post
        self.com = pre.columns[0]
        self.rate = rate

        # TODO: Implement **kwargs input dict for models?

        self.split_test_train()
        self.Model = smf.ols(self.com + '~' + params, data = self.train)
        self.Fit = self.Model.fit()

        #Make predictions called "Calcs"
        self.trainCalc = self.Fit.predict(self.train)
        self.testCalc = self.Fit.predict(self.test)
        self.postCalc = self.Fit.predict(post)

        # XXX: does this work? what is fit.resid?
        self.postDiff = self.postCalc - self.post[self.com]
        self.postCumsum = self.postDiff.cumsum()[-1]
#        self.savings = self.Fit.resid * self.rate

        ## CHECK
        # Need to add savings
        # need to savings rate


        #    r2 = r2_score(mnv.post[mnv.com], mnv.postModel)
        #    print(r2)
        #    mse = mean_squared_error(mnv.post[mnv.com], mnv.postModel)
        #    print(mse)

        #    aR2 = 1-(1-r2)*(len(mnv.post)-1)/(len(mnv.post)-14-1) #14 is len params
        #    print(aR2)
        #     cvrmse = sqrt(mse) / (len(data)-len(params) / mean(y)
        #    cvrmse = math.sqrt(mse) / (len(mnv.post)-14) / mnv.post.mean()

        ## Check cvrmse?

    def split_test_train(self, how='simple'):
        """
        Take the pre data set and create two variables "test" and "train" to
        feed into the model

        Split the data in the following ways:
            -1/3 : 2/3 simple split
            -a random collection of 1/3 : 2/3
            -as a Kfold with or without shuffling
        """

        how = how.lower()
        length = len(self.pre)

        if how == 'simple':
            twoThirds = int(round(length * 2/3))
            self.train = self.pre.iloc[0:twoThirds]
            self.test = self.pre.iloc[twoThirds:length]

        elif how == 'random':
            #TODO: Program this
            pass


    def model_plot(self):

        fig = plt.figure(figsize=(figW, figH*1.5))
        ax0 = plt.subplot2grid((2,1), (0,0))

        ax1 = plt.subplot2grid((2, 1), (0, 0), colspan=1)
        self.test[self.com].plot(label='actual')
        self.testCalc.plot(label='model')

        plt.title('Test data ' + self.params)
        plt.ylabel(self.com)
        plt.legend()

        ax2 = plt.subplot2grid((2, 1), (1, 0), colspan=1)

        self.post[self.com].plot(label='actual')
        self.postCalc.plot(label='model')
        plt.ylabel(self.com)
        plt.title('Post data ' + self.params)
        plt.legend()
        plt.tight_layout()

    # TODO: finish coding this function
    def stats_plot(self):

        fig = plt.figure(figsize=(figW*1, figH * 0.75))
        ax0 = plt.subplot2grid((1,3), (0, 0))

        ax1 = plt.subplot2grid((1,3), (0, 0))
        sm.qqplot(self.Fit.resid, ax=ax1)

        ax2 = plt.subplot2grid((1,3), (0, 1))
        ax2.scatter(self.post[self.com].values,
                    self.postCalc.values,
                    s=1, c = 'k')
        plt.axis('equal')
        plt.xlabel('Post actual')
        plt.ylabel('Post Calc')

        ax3 = plt.subplot2grid((1,3), (0, 2))
        ax3.scatter(self.post[self.com].values,
                    self.postDiff.values,
                    s=1, c = 'k')
        plt.axis('equal')
        plt.xlabel('Post actual')
        plt.ylabel('Post Calc Residuals')
        plt.tight_layout()


    def savings_plot(self, yaxis='raw', rate=1):

        if yaxis == 'raw':
            ydata = self.postDiff
            ylab = '[' + self.com + ']'

        elif yaxis == 'dollars':
            ydata = self.postDiff * rate
            ylab = '[$]'

        else:
            raise ValueError('savings_plot requires y-axis to be == raw or dollars')
            return


        fig = plt.figure(figsize=(figW*1.2, figH*1.5))
        ax0 = plt.subplot2grid((2,1), (0,0))

        ax1 = plt.subplot2grid((2, 1), (0, 0), colspan=1)

        savingsPos = ydata[ydata >= 0]
        savingsNeg = ydata[ydata < 0]

        plt.plot(savingsPos, color = 'k', linestyle='',marker='.', markersize=4)
        plt.plot(savingsNeg, color = 'r',linestyle='',marker='.', markersize=4)
#        self.postTest.plot(label='model')
        plt.title('Savings predicted by ' + self.params)
        plt.ylabel('Savings {}'.format(ylab))
        plt.legend()

        ax2 = plt.subplot2grid((2, 1), (1, 0), colspan=1)

        cumulative = ydata.cumsum()

        cumPos = cumulative[cumulative >= 0]
        cumNeg = cumulative[cumulative < 0]

        plt.plot(cumPos, color = 'k', linestyle='',marker='.', markersize=4)
        plt.plot(cumNeg, color = 'r', linestyle='',marker='.', markersize=4)

#        self.savings.cumsum().plot(label='savings cumulative')
#        self.prediction.plot(label='model')
#        plt.title('Savings predicted by ' + self.params)
        plt.ylabel('Cumulative Savings')
        plt.legend()
        plt.show()



    @staticmethod
    def k_test(pre, post, com, params, folds = 4):
        """
        Split data in folds number of folds
        run linear model on large slice
        make prediction of remaining slice

        compare all combinations

        ensure they're pretty consistent
        report flags


        ## CHeck pretty rough, ask down how Kfold should be used (before/after all models?)
        """

        kf = KFold(n_splits = folds, shuffle=True)

        for train_index, test_index in kf.split(pre):
            train, test = pre.iloc[train_index,:], pre.iloc[test_index,:]

            model = smf.ols(com + params, train).fit()
            predicted = model.predict(test)

#            print()

            print(mean_squared_error(test[com], predicted))
            print('')
            print(model.summary())




    def show():
        """
        Will plot data in many different ways
        "raw"
        "clean"
        "pre"
        "post"
        """

        if showWhat == 'clean':
            # plot the cleaning step again
            pass

        if showWhat == 'pre':
            # plot the pre dat
            pass

        # Etc...

#==============================================================================
# TESTS
#==============================================================================

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
             'sliceType' : 'ranges', #half, middate, ranges
             'midDate' : '2018-03-01', #only needed with sliceType : 'middate'
             'dateRanges' : ['2016-01-01','2017-01-01','2017-01-02','2018-01-01'],
             'OATsource' : 'file', 'OATname' : None}

    dc = data_handler(data[data.columns[0]].to_frame(), inputDict)
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
    dc._resample()
    dc.remove_outliers()
    dc.add_degree_hours()
    dc.data_slice()

    return dc

def mc_test_single(data):
    """

    """
    ## INPUT PARAMS
    inputDict = {'IQRmult' : 4.0,
                 'IQR' : 'y',
                 'resampleRate' : 'D',
                 'verbosity' : 3,
                 'sliceType' : 'ranges', #half, middate, ranges
                 'midDate' : '2018-03-01', #only needed with sliceType : 'middate'
                 'dateRanges' : ['2016-05-01','2016-08-01','2016-08-02','2018-01-01'],
                 'OATname' : None,
                 'OATsource' : 'file'} #only needed with sliceType : 'ranges'}

    dc = data_handler(data, inputDict)
    dc.default_clean()

    mod1 = ols_model(dc.pre, dc.post, 'CDH + HDH + C(weekend)')
#    mod1.split_test_train(how='simple')
    mod1.model_plot()
#    mod1.savings_plot(yaxis='dollars')
    mod1.stats_plot()
#
    return mod1, dc

def mc_test_all_linear(data):
    """
    K fold test
    """
    ## INPUT PARAMS
    inputDict = {'IQRmult' : 4.0,
                 'IQR' : 'y',
                 'resampleRate' : 'H',
                 'verbosity' : 3,
                 'sliceType' : 'half', #half, middate, ranges
                 'midDate' : '2018-03-01', #only needed with sliceType : 'middate'
                 'dateRanges' : ['2016-01-01','2017-01-01','2017-01-02','2018-01-01'],
                 'OATname' : None,
                 'OATsource' : 'file'} #only needed with sliceType : 'ranges'}

    dc = data_handler(data, inputDict)
    dc.default_clean()

    mc = many_ols(dc.pre, dc.post)
    mc.run_all_linear()

    return mc, dc


if __name__ == "__main__":

    filePath = 'data/arc2yeardata.xlsx'
    data = pd.read_excel(filePath,header=0, index_col=0, parse_dates=True)
#    dc0 = dc_test_init(data)
#    dc1 = dc_test_cleaning(data)
    mod1, dc1 = mc_test_single(data[data.columns[1]])
#    mc2, dc2 = mc_test_all_linear(data[data.columns[1]])

    pass