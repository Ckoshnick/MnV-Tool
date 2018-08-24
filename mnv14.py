# -*- coding: utf-8 -*-
"""
Created on Wed Apr 04 10:07:41 2018

@author: koshnick

Last update 5/14/18 - koshnick
Added Params subclass to data keeper. Refactored code to fit new convention
Next time - need to add internal functions to _convention.

v1.2 6/12/2017

- remove postCalc graph from model_plot() since this plot is not needed to eval
    uate the quality of the model. postCalc plot now shown in savings plot.
- added cvrmse stat as mod.Fit.cvrmse
- removed MSE from stats_pool (in favor of cvrmse)
- added calculate_vif function and stored result under mod.vif
- Fixed the ability to use OAT within the same dataset, however it is clunky
- Added custom markersize variable for savings plot
- Added ModelParameters class to allow for a modelDict input for the ols_model
    class

known issues:
- Something in the cleaning step or outlier plotting has become very slow and
    needs optimizing [I think its the OAT loader, has needed help anyways]
- coloration for plots is lacking
- Something happening where "CDH" or "HDH + CDH" wont run without a time var
- Imrpove OAT loader, and self OAT data

v1.3 6/25/2017
- updated plot colors / options / formatting
- updated docstrings
- resolved issue of slow OAT data loading in mypy library
- resolved issue of CDH failing with no time variable
- Added Kfold function and two reporting styles (absolute MSE and relative)

v1.3 6/28/2017

ols_model changes:
- Fixed deprecation warning for calculate_degree_days()
- changed old paramList to paramPermuteList (for many_ols)
- added paramList and paramString to ols_model
- added function to convert paramString to paramList and vica versa
- updated inputDict to reflect above changes
- updated VIF functinon to use paramList which simplifies function greatly
- removed get_columns_from_params() as this function is now obsolete
- removed get_dummy_strings() as this function is now obsolete
- created _remove_degree_days()
- removed function k_test() as it was unfinished and unused
- removed function make_param_string() as this function is now obsolete
- removed function show() as it was unfinished and unused
- Fixed issue where calculate_vif() would throw error
    "MissingDataError: exog contains inf or nans" .The OAT data was creating
    HDH/CDH values which were nan and causing this error

v1.4 8/__/2018
- Updated Kfold and split_test_train so now the train and test for the main
    model are identical to fold 0 in the Kfold analysis. Added variable
    self._folds that is a collection of indicies generated from sklearn KFold()
    This is used to now generate test_train, and reused to calc the Kfold stats
- Fixed conversion of paramList to paramString in ols_model's
     __init__. It was formerly dropping HDH CDH HDH2 and CDH2
- Changed _outliter_plot to be similar to other plotting functions where the
    outlier data is now stored in its own variable of the data keeper class
    so that it can be called upon later, and not within the remove_outlier
    function

- added resampled_plot() so that the workaround to use the old _outlier_plot()
    as a method to see the data after resampling is no longer needed. Just
    call resampled_plot() to see the data being fed into data_slice()
- made a change so that all plot objects are stored as self. <variable>
    so they plots can be recalled to be saved in the archiving function
- added try - except clauses to functions where dict.iteritems() for py27 was
    causing conflict in py36. Now the except class uses dict.items() for py36


-Added archiving feature to the parent class. This archive saves the following

    As an excel file:
    results of the VIF caluculation
    kfold calculations
    mod.Fit.summary() object

    As an excel file:
        dk.rawData - for loading into __future model program__ #TODO: name me

    As .png files:
    relevant plots for reporting

    As a .json:
    parameters to re-run models

    needs:
        - saving all data as a pdf report
        - a way to communicate with the larger archive

- Renamed the data_handler class to data_keeper (dk in future instances)
- Disabled the "undo" function since we're not using the CLI that was planned
- Removed date and name from data_keeper class since they were intended for
    archiving, but archiving is being done sperately now
- changed data_keeper functionality to allow any size dataframe to be passed in
    as the 'data' variable a new parameter is now needed to pass an index or
    column name to select the col that will be selected as self.ModifiedData
- Added the remodel class to allow saved models to be reloaded and updated with
    new PI data. Revisiaulzied, and saved
- Added work around in _resample function to make the issues where
    sum([nan, nan]) = 0 and mean([nan, nan]) = nan were causing issues that
    added several erroneous zeros into the dataset.
    This removes the following from known issues:
 X removing a full day of outliers will cause a value of 0 to appear when
    that day is resampled, becuase the date axis will be filled with the
    missing day, and the value will be set to zero. maybe resample has an
    option to change this behavior



known issues:
    - TODO: Fix doc strings containing "exhautive numpydoc format docstring."
    - Models are limited to "D" or "H" resampled due to the implementation in
        how OAT is turned into degree days (eg can not do a 3hr long interval)



Last update 8/2/18 - koshnick
"""

import sys
import math
import json
#import mypy
#import pickle
import itertools

sys.path.append('../../mypy')

import numpy as np
import pandas as pd
import seaborn as sns

import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from os import mkdir
from datetime import datetime
#from PI_client2 import pi_client
from PI_Client.v2 import pi_client #latest verison of PI Client
from mypy import build_time_columns, calculate_degree_hours, merge_oat

from sklearn.model_selection import KFold
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
#from sklearn.metrics import mean_squared_error, r2_score

#path_prefix = path.dirname(path.abspath(__file__))
plt.rcParams.update({'figure.max_open_warning': 0})
sns.set()  # Enable to turn on seaborn plotting
pi = pi_client()

figW = 18
figH = 6

version = 'Version 1.4'


class DataParameters():
    """
    A simple class to store attributes for the data_keeper class. The __init__
    function is specially formated for the data_keeper class and takes in a
    **kwargs dictionary to populate the parameters. All parameters have default
    values, but can be modified in **kwargs.

    Parameters
    ----------
    **inputDict : dict
        each entry in the dictionary should map to a variable in the __init__

    inputDict = {
         'column': 0
         'IQRmult' : 3.0,
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

    column : int or str
        if str - name of column to select if int - column index
    IQRmult : float
        Number that the IQR will be bultiplied by to adjust floor/ceiling
    IQR : str [either 'y' or None]
        if IQR == 'y' then the outlier detection will use IQR method
    floor : int
        Lowest value any data point should take on

    #TODO: Finish writting docstring

    Returns
    -------
    class instance
        instance.attibutes

    Raises
    ------
    None
    """

    def __init__(self,
                 column = 0,
                 IQRmult=3,
                 IQR='y',
                 modifiedDataInterval = None,
                 floor=-1,
                 ceiling=10000,
                 resampleRate='D',
                 sliceType='half',
                 midDate=None,
                 dateRanges=None,
                 OATsource='file',
                 OATname=None):

        self.column = column
        self.modifiedDataInterval = None
        self.OATsource = OATsource
        self.floor = floor
        self.ceiling = ceiling
        self.IQR = IQR
        self.IQRmult = IQRmult
        self.resampleRate = resampleRate
        self.sliceType = sliceType
        self.midDate = midDate
        self.dateRanges = dateRanges
        self.OATname = OATname

    def show_params(self):
        """ Display the key, value pairs of the parmeters in this class"""
        for k, v in self.__dict__.iteritems():
            print(k, v)


class data_keeper():
    """
    Multipurpose data cleaning class. Designed to take in a pd.DataFrame,
    modify it in several ways, cut the data, and pass it off for MnV modeling.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Timeseries data
    inputDict : dict
        **kwargs type dict to be passed into the DataParameters class

    Returns
    -------
    instance: dk
        notable attributes
        ------------------
        dk.modifiedData - working data object
        dk.restoreData - a copy made before modifiedData is changed
        ## TODO: Should I make an option to shut this off to increase speed
            # Currently shutoff
        dk.pre - pre period data created by data_slice()
        dk.post - post period data created by  data_slice()


    Raises
    ------
    ## TODO: Write me
    KeyError
        when a key error
    OtherError
        when an other error
    """

    # Data Class init
    # TODO: Allow a dataframe of N columns to be passed in,
    # TODO: Create method to specify which column to choose for data- default 0

    def __init__(self, data, inputDict=None):
        # instantiate params
        self.params = DataParameters(**inputDict)
        self.params.modifiedDataInterval = 'raw'

        # Allow for pd.series or pd.dataframe to be loaded. In the case of a
        # dataframe, use the first column as the data of interest, drop others

        if isinstance(data, pd.Series):
            self.rawData = data.to_frame()

        self.rawData = data.copy()

        # TODO: rename self.com to something more meaningful
        self._set_column(self.params.column)

        # Use the OAT loaded into instance if present, otherwise OAT data will
        # be loaded from OAT master
        if self.params.OATsource == 'self':
            self.OAT = data[self.params.OATname]

        # modifiedData is the working variable for all data modifications
        self.modifiedData = self.rawData[self.com].to_frame()

    def undo(self):
        """ Allows the user to go back one step in data modification """
        self.modifiedData = self.restoreData.copy()


# =============================================================================
# CLEANING
# =============================================================================

    def _set_column(self, column):
        """ Handles the column input during the __init__ of data_keeper"""
        if isinstance(column, int):
            self.com = self.rawData.columns[column]
        elif isinstance(column, str):
            try:
                assert(column in list(self.rawData.columns))
            except AssertionError:
                print('input param "column" must be in "data.columns"')
                print('{} not in {}'.format(column, self.rawData.columns))
            self.com = column

    def _IQR(self):
        """
        Calculates new boundaries based on the Inner Quartile Range multiplied
        by a IQR multiplier (default 3) ##CHECK
        IQRmult taken from object attribute

        Passes new upper and lower bounds to the remove_outliers. Boundarires
        change only if they are more restrictive than the parmas.floor and/or
        params.ceiling limits provided to remove_outliers

        Parameters
        ----------
        params.IQRmult : array_like
            multiplier for the IQR used to calculate new potential boundaries
        self.modifiedData :
            the 2nd param
        third : {'value', 'other'}, optional
            the 3rd param, by default 'value'

        Returns
        -------
        upper, lower : float, float
            New potential boundaries for remove_outliers

        Raises
        ------
        None
        """

        Q1 = self.modifiedData[self.com].quantile(0.25)
        Q3 = self.modifiedData[self.com].quantile(0.75)
        IQR = Q3 - Q1

        print('Q(75%): {0:.2f} Q(25%): {1:.2f}'.format(Q3, Q1))
        print('IQR value is {0:.2f}'.format(IQR))

        upper = round(Q3 + (self.params.IQRmult * IQR), 2)
        lower = round(Q1 - (self.params.IQRmult * IQR), 2)

        return upper, lower

    def remove_outliers(self,
                        floor=None,
                        ceiling=None,
                        IQR=None):
        """
        Tests if data is between bounds (floor, ceiling) and removes entries
        that are out of bounds

        Parameters
        ----------
        floor : int
            replaces params.floor if provided. Represents the lowest value the
            data should take on.
        ceiling : int
            replaces params.ceiling if provided. Represents the max value the
            data should take on.
        IQR : Bool
            If True perform the _IQR function and change boundaries
            [floor,ceiling], if _IQR is more restrictive. Otherwise just use
            [floor,ceiling]

        Returns
        -------
        None
            modifies attribute self.modifiedData
            _outlier_plot

        Raises
        ------
        None
        """

        # replace params.X if supplied to function
        if floor == None: floor = self.params.floor
        if ceiling == None: ceiling = self.params.ceiling
        if IQR == None: IQR = self.params.IQR

        # Store copy incase self.undo() invoked
        # self.restoreData = self.modifiedData.copy() #disabled july 31,18 v1.4

        # temp needed for being temporary modifiedData
        temp = self.modifiedData.copy()

        if IQR:
            IQRupper, IQRlower = self._IQR()
            print('IQRupper', IQRupper, 'IQRlower', IQRlower)

            if IQRupper < ceiling:
                ceiling = IQRupper
                print('Ceiling adjusted by IQR - Now {0:.2f}'.format(ceiling))
            if IQRlower > floor:
                floor = IQRlower
                print('Floor adjusted by IQR   - Now {0:.2f}'.format(floor))

        # Select data where floor < data < ceiling
        if floor != None:
            temp = temp.where(temp > floor, other=np.nan)
        if ceiling != None:
            temp = temp.where(temp < ceiling, other=np.nan)
        else:
            pass


        indexDifference = self.modifiedData.index.difference(temp.dropna().index)

        self.outliers = self.modifiedData[self.com][indexDifference]
        self.modifiedData[self.com] = temp
#        self.modifiedData.dropna(inplace=True)

    def add_oat_column(self):
        """
        Call mypy to add OAT data to the commodity data. Only need to call this
        if the input data was not supplied with assocaited OAT data.
        """
        # Store copy incase self.undo() invoked
        # self.restoreData = self.modifiedData.copy() #disabled july 31,18 v1.4

        if self.OATsource == 'file':
            # TODO: rework MYPY and this function
            self.modifiedData = merge_oat(self.modifiedData,
                                          choose='y')

        elif self.OATsource == 'self':
            self.modifiedData['OAT'] = self.OAT

        print('Created OAT from {}'.format(self.OATsource))
        print('')

#        return self ##CHECK Do i even need this?

    def _resample(self, resampleRate=None, aggFun='mean'):
        """ Calls resample rate pandas native with default argument handling"""

        # self.restoreData = self.modifiedData.copy() #disabled july 31,18 v1.4
        # TODO: Fix issue where resampling an empty day gives a 0 value.
        # Just drop zeros?


        if resampleRate == None:
            resampleRate = self.params.resampleRate


        # TODO: Remove me if not used
        keepIndex = self.modifiedData.resample(resampleRate).mean().dropna()

        self.modifiedData = self.modifiedData.resample(resampleRate).agg(aggFun)
        self.modifiedData = self.modifiedData.loc[keepIndex.index]

        self.params.modifiedDataInterval = resampleRate

    def add_degree_hours(self, cutoff=65):
        """
        Use OAT data to generate CDD and HDD. In this implementation we sample
        the HD-hours and CD-hours and then sum them up to what ever timestep
        we're using for the M&V (usually daily)

        Parameters
        ----------
        cutoff : integer default = 65
            the cutoff temp of when we move from 'heating' to 'cooling'

        Returns
        -------
        self
        modifies self.modifiedData to have 4 new columns HDH, CDH, CDH2, HDH2

        Raises
        ------
        None
        """
        # self.restoreData = self.modifiedData.copy() #disabled july 31,18 v1.4

        # TODO: Should not use resample rate here, should use actual invterval
        # assert that it is H or D??

        if self.params.OATsource == 'file':
            hours = calculate_degree_hours(oatData=None,
                                           by=self.params.resampleRate,
                                           cutoff=cutoff)

        elif self.params.OATsource == 'self':
            hours = calculate_degree_hours(oatData=self.OAT,
                                           by=self.params.resampleRate,
                                           cutoff=cutoff)

        self.modifiedData['HDH'] = hours['HDH']
        self.modifiedData['CDH'] = hours['CDH']

        # TODO:. How do we sum HDH2 vs HDD2?
        self.modifiedData['HDH2'] = self.modifiedData['HDH'] ** 2
        self.modifiedData['CDH2'] = self.modifiedData['CDH'] ** 2

        # Drop any row where null values appear
        # TODO: Make sure Nans arebeing handled properly by HDH and resample. some data sets seem to be losing much data
#        self.modifiedData = self.modifiedData.dropna(axis=0, how='any')

    def add_time_columns(self):
        """ Calls the mypy function build_time_columns """
        # TODO allow the specification of nighttime daytime hours?
        self.modifiedData = build_time_columns(self.modifiedData)

    # TODO: Commented this function out to see if it would break the program
    # Need to remove it from DK class since it is not used. Dummies are made
    # in the model class
#    def add_dummy_variables(self):
#
#
#        dummyColumns = ['month','weekday','dayofweek','hour']
#        dums = pd.get_dummies(self.modifiedData[dummyColumns],
#                              columns=dummyColumns,
#                              drop_first=True)
#
#
#        self.modifiedData = pd.concat([self.modifiedData,dums],axis=1)
#        pass

    def data_slice(self):
        """
        This function seperates the pre and post period data by slicing it in
        one of three ways specified by sliceType
        how = 'half'- split the data 50/50
        how = 'middate' - split the data at the date of middate
        how = 'ranges' - split the data in to [pre1:pre2] and [post1:post2]

        example datestring: "2017-02-28"

        Parameters
        ----------
        None: specified in "params"

        Returns
        -------
        self
        modified self.pre and self.post inplace

        Raises
        ------
        ValueError
            When sliceType is not supplied with the slice date(s)
        """

        how = self.params.sliceType

        if how == 'half':
            dataLength = len(self.modifiedData)
            midPoint = int(math.floor(dataLength)/2)

            self.pre = self.modifiedData.iloc[0:midPoint, :]
            self.post = self.modifiedData.iloc[midPoint:, :]

        elif how == 'middate':
            if self.params.midDate:
                midDate = pd.to_datetime(self.params.midDate)

                # iloc to avoid duplicating the point at middate
                self.pre = self.modifiedData[:midDate].iloc[:-1]
                self.post = self.modifiedData[midDate:]
            else:
                raise ValueError('data_slice: Must supply midDate with how = '
                                 '"middate"')

        elif how == 'ranges':
            if self.params.dateRanges:
                ranges = self.params.dateRanges  # TODO check lazy spaghetti
                preStart = pd.to_datetime(ranges[0])
                preEnd = pd.to_datetime(ranges[1])
                postStart = pd.to_datetime(ranges[2])
                postEnd = pd.to_datetime(ranges[3])

                self.pre = self.modifiedData[preStart:preEnd]
                self.post = self.modifiedData[postStart:postEnd]
            else:
                raise ValueError('data_slice: Must supply datranges for how '
                                 ' = "ranges"')

    def default_clean(self):
        """
        A suite of data processing functions. This should be sufficient for
        99% of models generated with the MnV tool

        #TODO: Rename this function to something more descriptive since its not
        only cleaning

        Ordering:
            1. Remove outliers and leave as raw data
            2. Resample data to 1H interval
            3. Resample data to self.resampleRate
            4. Use OAT from file or from self to calculate HDH and CDH
            5. use mypy.build_time_cols to create time columns
            6. Slice data into pre/post
        """

        self.remove_outliers()
        self._resample(resampleRate='H', aggFun='mean')
        self.modifiedData = self.modifiedData.dropna()
        self._resample(aggFun='sum')

        self.add_degree_hours()
        self.add_time_columns()
        self.data_slice()


# =============================================================================
# PLOTTING
# =============================================================================

    def _outlier_plot(self, yrange=None, title=None):
        """
        This plotting method allows the user to see which outliers are being
        removed from the dataset, and a statistics comparison in the form of a
        boxplot. Red datapoints indicated points to be removed.

        Parameters
        ----------
        yrange : tuple
            if wanting to constrain the yrange to (ymin,ymax)
        title : str
            custom title string -- probably useless

        Returns
        -------
        None - saves plot as class variable

        Raises
        ------
        None

        """

        # setup vars
        widthFactor = 1.0
        heightFactor = 1.0

        fig = plt.figure(figsize=(figW * widthFactor, figH * heightFactor))
        ax0 = plt.subplot2grid((1, 5), (0, 0))

        # Box plot
        ax1 = plt.subplot2grid((1, 5), (0, 0), colspan=1)

        indexDifference = self.rawData.index.difference(self.outliers)

        noOutliers = self.rawData[self.com][indexDifference]

        _comb = pd.concat([self.rawData[self.com],
                          noOutliers],
                          axis=1)
        _comb.columns = ['before', 'after']
        sns.boxplot(data=_comb, ax=ax1)
        plt.ylabel(self.com)
        plt.title('Before and After')

        # Scatter plot
        ax2 = plt.subplot2grid((1, 5), (0, 1), colspan=4)

        plt.plot(noOutliers, color='k', linestyle='', marker='.')
        plt.plot(self.outliers, color='r', linestyle='', marker='.')

        plt.title('Outlier removal result. interval = raw'.format(
                str(self.params.modifiedDataInterval)))

#        self.params.modifiedDataInterval = resampleRate
        if yrange:
            plt.ylim(yrange)
        plt.show()

        self.outlierPlot = ax0

    def _resampled_plot(self, yrange=None, title=None):
        """
        This plotting method allows the user to see the data after outliers are
        removed and the final resampling is done.

        Parameters
        ----------
        yrange : tuple
            if wanting to constrain the yrange to (ymin,ymax)
        title : str
            custom title string -- probably useless

        Returns
        -------
        None - saves plot as class variable

        Raises
        ------
        None

        """
        # setup vars
        widthFactor = 1.0
        heightFactor = 1.0

        fig = plt.figure(figsize=(figW * widthFactor, figH * heightFactor))
        ax0 = plt.subplot2grid((1, 5), (0, 0))

        # Box plot
        ax1 = plt.subplot2grid((1, 5), (0, 0), colspan=1)

        sns.boxplot(data=self.modifiedData[self.com], ax=ax1)
        plt.ylabel(self.com)
        plt.title('clean & resampled data')

        # Scatter plot
        ax2 = plt.subplot2grid((1, 5), (0, 1), colspan=4)

        plt.plot(self.modifiedData[self.com],
                 color='k', linestyle='', marker='.')

        plt.title('Outlier and resample removal result. interval = {}'.format(
                str(self.params.modifiedDataInterval)))

        if yrange:
            plt.ylim(yrange)
        plt.show()

    def _pre_post_plot(self):
        """
        This plotting method allows the a line plot of the data after has been
        cut into pre and post sections

        Parameters
        ----------
        None

        Returns
        -------
        None - saves plot as class variable

        Raises
        ------
        None

        """
        # setup vars
        widthFactor = 1.0
        heightFactor = 1.0

        fig = plt.figure(figsize=(figW * widthFactor, figH * heightFactor))
        ax0 = fig.add_subplot(111)

        ax0.plot(self.pre[self.com], color='royalblue')
        ax0.plot(self.post[self.com], color='forestgreen')
        plt.legend(['pre', 'post'])
        plt.title('Pre and Post period')
        plt.ylabel(self.com)
        plt.show()

        self.prepostPlot = ax0


# =============================================================================
# MODELING
# =============================================================================

class ModelParameters():
    """
    A simple class to store attributes for the ols_model class. The __init__
    function is specially formated for the data_keeper class and takes in a
    **kwargs dictionary to populate the parameters. All parameters have default
    values, but can be modified in **kwargs.

    Parameters
    ----------
    **inputDict : dict
        each entry in the dictionary should map to a variable in the __init__

    inputDict = {'params' : 'C(month) + C(weekday)',
         'testTrainSplit' : 'random',
         'randomState': 4291990,
         'testSize': 0.2,
         'commodityRate' : 0.056,
         'paramList' : ['','C(month)','C(weekday)']}

    params : string
        All of the params to run the ols model with
    testTrainSplit : string
        'random' - for a random test selection
        'simple' - Split the data in sequential order where train is the first
            segment
    randomState : int / None
        for 'random' testTrainSplit this is the seed for the randomlist
        specify an integer if you would like it to be static betweeen runs
    testSize : float [0.0 - 0.99]
        The ratio of data used in the test set
    commodityRate : float
        The price in dollars for the selected commodity
    paramList : list of strings
        All variables that will be ran in many_ols.test_all_linear()


    Returns
    -------
    class instance
        instance.attibutes

    Raises
    ------
    None
    """

    def __init__(self,
                 params=['month', 'weekday'],
                 paramList=None,
                 paramString=None,
                 testTrainSplit='random',
                 randomState=4291990,
                 testSize=0.2,
                 commodityRate=0.056,
                 paramPermuteList=['', 'C(month)', 'C(weekday)']):

        # TODO Will be changed to not None with _convert_params
        self._convert_params(params)

        # self.params = params #Goodbye params
        self.testTrainSplit = testTrainSplit
        self.randomState = randomState
        self.testSize = testSize
        self.commodityRate = commodityRate
        self.paramPermuteList = paramPermuteList

    def _convert_params(self, params):
        """
        Convert the params #TODO: call them vars? from the list form ['month','hour']
        to the string form 'C(month) + C(hour)' or vice versa. Important for
        allowing flexibility in how the vars are input and it helps the build
        dummy section play nice.

        Parameters
        ----------
        params : string or list
            All of the params to run the ols model with

        Returns
        -------
        list or string
            if list input, string returns. if string input list returns

        Raises
        ------
        ValueError
            Input params must be list or string
        """

        # TODO: Rename model params to vars and keep params for the dict object namespace?

        DD = ['HDH', 'CDH', 'CDH2', 'HDH2']
        if isinstance(params, list):
            self.paramList = list(params)
            newParams = []
            for param in self.paramList:
                if param in DD:
                    newParams.append(param)
                else:
                    newParams.append("C({})".format(param))

            paramString = ' + '.join(newParams)
            self.paramString = paramString

            # conver to string, paramList
        elif isinstance(params, str):
            paramList = params.split('+')
            paramList = [x.replace("C(", "").replace(")", "").strip(' ')
                         for x in paramList]
            self.paramList = paramList
            self.paramString = params

            # convert to list, paramString
        else:
            raise ValueError('Params input must be either string or list as '
                             ' form ["month","hour"] or "C(month) + C(hour)"')

    def show_params(self):
        """ Display the key, value pairs of the parmeters in this class"""
        for k, v in self.__dict__.iteritems():
            print(k, v)


class ols_model():
    """
    Designed to take inputs from data_keeper and a **kwargs inputDict
    Performs a single linear regression based on input params, calculates
    predicted values using the self.post data set, and reports statistics
    information about the model

    Parameters
    ----------
    pre : pd.DataFrame
        Timeseries data passed from data_keeper
    post : pd.DataFrame
        Timeseries data passed from data_keeper
    inputDict : dict
        **kwargs type dict to be passed into the ModelParameters class

    Returns
    -------
    instance: mc
        notable attributes
        ------------------
        ## TODO: fill me

    Raises
    ------
    ## TODO: Write me
    KeyError
        when a key error
    OtherError
        when an other error
    """

    def __init__(self, pre, post, inputDict):

        self.params = ModelParameters(**inputDict)
        self.pre = pre
        self.post = post
        self.com = pre.columns[0] # TODO: Allow user to specify this
        self.dataInterval = pd.infer_freq(self.pre.index)

        self.split_test_train(how=self.params.testTrainSplit)

        self.Model = smf.ols(self.com + '~' + self.params.paramString,
                             data=self.train)
        self.Fit = self.Model.fit()

        self.Fit.cvrmse = math.sqrt(self.Fit.mse_resid) / self.train[self.com].mean()

        #Make predictions called "Calcs"
        self.trainCalc = self.Fit.predict(self.train)
        self.testCalc = self.Fit.predict(self.test)
        self.postCalc = self.Fit.predict(post)

        self.postModel = smf.ols(self.com + '~' + self.params.paramString,
                                 data=self.post)

        #TODO: WHy is this commented out? looks like it needs deleting
#                    try:
#                self.inputDict['params'] = params
#                outputs[params] = ols_model(self.pre, self.post, self.inputDict)

        #TODO: make sure workflow includes calculating VIF outside of the __init__

#        try:
#            self.calculate_vif()
#        except Exception as e:
#            print('Could not calculate VIF for {}'.format(self.params.paramString))
#            print('Exception caught: {}'.format(e))
#            self.vif = None

        self.trainDiff = self.trainCalc - self.train[self.com]
        self.postDiff = self.postCalc - self.post[self.com]
        self.postCumsum = self.postDiff.cumsum()[-1]

    def _remove_degree_days(self, params):
        """This removes and reinstalls the degree days for dummy creation"""

        params = list(params)  # Ghetto copy
        dumList = ['CDH', 'HDH', 'CDH2', 'HDH2']
        removed = []

        for dum in dumList:
            if dum in params:
                params.remove(dum)
                removed.append(dum)
            else:
                pass

        return params, removed

    def _infer_interval(idx):
        pass




    def calculate_vif(self):
        """
        Variance Inflation Factor (VIF) is a method to determine if two input
        variables are highly correlated with one another. This is something to
        avoid in models. Typically values less than 5 are appropriate.

        Parameters
        ----------
        self.pre : pd.DataFrame
            This calculation is run on the pre dataset

        Returns
        -------
        pd.DataFrame
            results of VIF caluculation paired with the dummy variables is
            returned as well as stored in self.vif

        Raises
        ------
        None

        """

        #TODO: IF VIF HAS TOO FEW COLUMNS THEN ABORT
        # Get dummies
        dummyColumns, removed = self._remove_degree_days(self.params.paramList)

        try:
            dums = pd.get_dummies(self.pre[dummyColumns],
                                  drop_first=True,
                                  columns=dummyColumns)
        except ValueError:
            dums = None
        else:
            dums = dums
        finally:
            self.dums = dums

        # join dummies to commodity and degree days
        testdf = pd.concat([self.pre[removed], dums], axis=1)
        # Add constant column if needed (essentially this is the intercept)

#        print(testdf.columns)
#        for dum in dums:
#            if len(testdf[dum].unique()) == 1:
#                del testdf[dum]
#        print(testdf.head())

        testdf = add_constant(testdf)  # VIF Expects a constant column

        # Get VIF values
        vif_values = [variance_inflation_factor(testdf.values, i)
                      for i in range(testdf.shape[1])]
        # Format into nice pd.DataFrame
        vif = pd.DataFrame(index=testdf.columns,
                           data=vif_values,
                           columns=['VIF'])

        self.vif = vif

        return vif

    def calculate_kfold(self):
        """
        Ensures that the slicing of data is not influencing the results. Kfold
        analysis will split the data into X cominations of test/train where
        X = 1/(test_size_fraction). So if test_size_fraction = 0.2 then 5 kfold
        splits will be made. Each time the testsize will be 0.2 and the train
        data set will be the remainder.

        This functions reports the kfold results as a relative deviation from
        the average of the mean squared error of all of the folds.

        explained again:
        1. Generate X fold
        2. Fit model params to all folds
        3. Calculate MSE for each model
        4. Average MSE for all folds
        5. Report the ratio of all folds individual MSE to the mean MSE

        Parameters
        ----------
        None

        Returns
        -------
        None
        pd.DataFrame
            results of KFold validation is stored in self.kfoldStats and
            self.kfoldRelative

        Raises
        ------
        None

        """

        # dict to hold final vales and transform to df
        statsPool = {}
        foldNumber = -1

        # _folds is a static set of indicies to re-references the slices
        for train_index, test_index in self._folds:

            foldNumber += 1

            # split into test/train based on kfold indicies
            train = self.pre.iloc[train_index, :]
            test = self.pre.iloc[test_index, :]

            # in this function to maintaine consitency
            Fit = smf.ols(self.com + ' ~ ' + self.params.paramString, train).fit()

            # TODO: DO we need to be using predicted or Fit? MATH IS HARD
#            predicted = Fit.predict(test)

            newStatsRow = {'R2' : Fit.rsquared,
                           'AR2' : Fit.rsquared_adj,
#                           'cvrmse': Fit.cvrmse,
                           'mse': Fit.mse_resid,
#                           'postDiff' : Fit.postCumsum,
                           }

            statsPool[foldNumber] = newStatsRow

        # Build DataFrame
        kfoldStats = pd.DataFrame(statsPool).T.sort_values('mse')
        # Re-order columns
        self.kfoldStats = kfoldStats[['R2', 'AR2', 'mse']]

        # Building relative MSE values
        mseMean = round(kfoldStats['mse'].mean())
        kfoldRelative = kfoldStats['mse'] / mseMean * 100
        kfoldRelative = kfoldRelative.append(pd.Series({'<mse>': mseMean}))

        self.kfoldRelative = kfoldRelative.rename('rel. pct.')

    def split_test_train(self, how='random', testSize=None, randomState=None):
        """
        Splits the pre period data into test and train sets for the model to be
        generated (train) with and evaluated against (test)

        Parameters
        ----------
        how : str default:'random'
            how determins which way the data will be sliced. [simple,random]
            simple - will just split the data at a single point where train is
            the begining of the data, and test the latter part
            random - shuffles the timeseries index and then splits the data
            into 1-testSize (train) and testSize (test) chunks
        testSize : float
            This determine which fraction (eg 0.25) of the pre dataset will be
            used in the test set. The remainder will be used in the train set
        randomState: int
            When using how = 'random' this is the seed value by which the
            random shuffle is carried out. if you leave it as None, a new
            shuffle will be used each time the function is ran. By specifying a
            value you can use the same shuffle over and over.


        Returns
        -------
        None

            results of splitting generate two pd.DataFrame and are stored in
            self.test
            self.train

        Raises
        ------
        None

        """

        #TODO: add more split options?

        if testSize == None: testSize = self.params.testSize
        if randomState == None: randomState = self.params.randomState

        how = how.lower()
        length = len(self.pre)

        if how == 'simple':
            # Must take 1-testsize, since testsize is the small fraction
            splitIndex = int(round(length * (1-testSize)))

            self.train = self.pre.iloc[0:splitIndex]
            self.test = self.pre.iloc[splitIndex:length]

        elif how == 'random':

            self._get_folds(randomState)

            self.train = self.pre.iloc[self._folds[0][0]]
            self.test = self.pre.iloc[self._folds[0][1]]

    def _get_folds(self, randomState):
        """ Splits the index into 1/(testSize) folds, stores in self._folds"""
        # Calculate # of folds from 1/testSize
        folds = int(round(1/self.params.testSize))

        # Use the scikitlearn kfolds function to generate randomized folds
        kf = KFold(n_splits=folds,
                   shuffle=True,
                   random_state=randomState)

        self._folds = [x for x in kf.split(self.pre)]

    def _get_tmy_data(self, startDate='pre', endDate='fiscal'):
        # find start/end dates
        # pull tmy data from pi
        # build hdh chd time columns and all that jazz

        currentTime = datetime.now()
        year = int(currentTime.year)
        month = int(currentTime.month)

        # Dealing with startDate
        if startDate == 'pre':
            startDate = self.pre.index[0].strftime('%Y-%m-%d')

        else:
            startDate = startDate

        # Dealing with endDate
        if endDate == 'endofyear':
            endDate = "{}-01-01".format(year+1)

        if endDate == 'fiscal':
            if month >= 7:
                year += 1
            else:
                pass
            endDate = "{}-07-01".format(year)

        elif len(endDate) == 4:
            try:
                if (int(endDate) - 2018) < 50:
                    year = int(endDate)
                    endDate = "{}-01-01".format(year+1)
            except ValueError:
                print('Cant convert {} to int'.format(endDate))
        else:
            endDate = endDate  # need code to check if it its 'yyyy-mm-dd'

        print(startDate, endDate)

        tmy = pi.get_stream_by_point(['Future_TMY'], start=startDate, end=endDate,
                                    interval = '1h')

        hours = calculate_degree_hours(tmy, by='hour')

        tmy['HDH'] = hours['HDH']
        tmy['CDH'] = hours['CDH']

        # TODO:. How do we sum HDH2 vs HDD2?
        tmy['HDH2'] = tmy['HDH'] ** 2
        tmy['CDH2'] = tmy['CDH'] ** 2

        self.tmy = tmy
        return tmy

    def _make_tmy_pre(self):
        """ Predict building behavior using tmy data """

        tmyPre = self.Fit.predict(self.tmy).to_frame()
        tmyPre.columns = ['tmyPre']

        self.tmyPre = tmyPre

    def _make_tmy_post(self):
        """ Predict building behavior using tmy data and post data"""

        tmyPost = self.postModel.fit().predict(self.tmy).to_frame()
        tmyPost.columns = ['tmyPost']

        self.tmyPost = tmyPost


    def compare_tmy_models(self, startDate='pre', endDate='fiscal'):
        # plot together the TMY PRE and the PRE period. calc RMSE [mind the interval]
        # plot goether the TMY POST and the POST period calc RMSE [mind the interval]
        # Calculate the TMY PRE and TMY POST diff [mind the interval] and find the cumsum and plot

        #TODO: Fix the plots!

        self._get_tmy_data(startDate=startDate, endDate=endDate)

        self.tmy = self.tmy.resample(self.dataInterval).sum()
        self.tmy = build_time_columns(self.tmy)

        self._make_tmy_post()
        self._make_tmy_pre()

        self.tmyPreComp = self.pre.merge(self.tmyPre, how='inner', right_index=True,
                                    left_index=True)
        self.tmyPreComp = self.tmyPreComp[[self.com, 'tmyPre']]
#        tmyPreComp = tmyPreComp.drop_time_columns()

        self.tmyPostComp = self.post.merge(self.tmyPost, how='inner',
                                      right_index=True, left_index=True)
        self.tmyPostComp = self.tmyPostComp[[self.com, 'tmyPost']]

        self.tmyFutureDiff = self.tmyPost['tmyPost'] - self.tmyPre['tmyPre']
        # compare distance? range, or typical year (days/365) * (365-days?)
        # Show in savings plot possible?

        pass


    def model_plot(self):
        """
        This plotting method allows the user to see the performance of the
        ols_model by viewing the actual/predicted test data and post data

        Parameters
        ----------

        Returns
        -------
        None - saves plot as class variable

        Raises
        ------
        None

        """

        _color = 'mediumorchid'
        _style = '--'

        widthFactor = 1.0
        heightFactor = 1.0

        fig = plt.figure(figsize=(figW * widthFactor, figH * heightFactor))
        ax0 = plt.subplot2grid((2, 1), (0, 0))

        ax1 = plt.subplot2grid((2, 1), (0, 0), colspan=1)

        plt.plot(self.test[self.com], label='actual', color='k')
        plt.plot(self.testCalc, label='model', color=_color, linestyle=_style)

#        self.test[self.com].plot(label='actual', color='k')
#        self.testCalc.plot(label='model', color=_color, linestyle=_style)

        plt.title('Test data ' + self.params.paramString)
        plt.ylabel(self.com)
        plt.legend()

        ax2 = plt.subplot2grid((2, 1), (1, 0), colspan=1)

        plt.plot(self.post[self.com], label='actual', color='k')
        plt.plot(self.postCalc, label='model', color=_color, linestyle=_style)
#
#        self.post[self.com].plot(label='actual', color='k')
#        self.postCalc.plot(label='model', color=_color, linestyle=_style)
        plt.ylabel(self.com)
        plt.title('Post data ' + self.params.paramString)
        plt.legend()
        plt.tight_layout()

        self.mod1Plot = fig

    def model_plot2(self):
        """
        This plotting method allows the user to see the performance of the
        ols_model by viewing the actual/predicted of JUST the test data

        Parameters
        ----------

        Returns
        -------
        None - saves plot as class variable

        Raises
        ------
        None

        """

        widthFactor = 1.0
        heightFactor = 0.75

        _color = 'mediumorchid'
        _style = '--'

        fig = plt.figure(figsize=(figW * widthFactor, figH * heightFactor))
        ax0 = plt.subplot2grid((1, 1), (0, 0))

        ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=1)

        plt.plot(self.test[self.com], label='actual', color='k')
        plt.plot(self.testCalc, label='model', color=_color, linestyle=_style)
#        self.test[self.com].plot(label='actual', color='k')
#        self.testCalc.plot(label='model', color=_color, linestyle=_style)

        plt.title('Test data ' + self.params.paramString)
        plt.ylabel(self.com)
        plt.legend()

        self.mod2Plot = fig

    def stats_plot(self):
        """
        This plotting method allows the user to see the performance of the
        ols_model by viewing 3 plots

        qqplot - shows if there is more or less fitting error for data values
        that are either larger or smaller

        calc vs actual - View the train data vs the calc train data
        resids vs actual - View the train data vs the calc resids of train data

        Parameters
        ----------
        None

        Returns
        -------
        None - saves plot as class variable

        Raises
        ------
        None

        """

        widthFactor = 1.0
        heightFactor = 0.75

        fig = plt.figure(figsize=(figW * widthFactor, figH * heightFactor))
        ax0 = plt.subplot2grid((1, 3), (0, 0))

        ax1 = plt.subplot2grid((1, 3), (0, 0))
        sm.qqplot(self.Fit.resid, ax=ax1)

        ax2 = plt.subplot2grid((1, 3), (0, 1))
        ax2.scatter(self.train[self.com].values,
                    self.trainCalc.values,
                    s=1, c='k')

        plt.axis('equal')
        plt.xlabel('Train actual')
        plt.ylabel('Train Calc')

        ax3 = plt.subplot2grid((1, 3), (0, 2))
        ax3.scatter(self.train[self.com].values,
                    self.trainDiff.values,
                    s=1, c='k')

        plt.axis('equal')
        plt.xlabel('Train actual')
        plt.ylabel('Train Calc Residuals')
        plt.tight_layout()

        self.statsPlot = fig

    def savings_plot(self, yaxis='raw', pointSize=4):
        """
        This plotting method allows the user to see the numeric or financial
        difference between the post and the postCalc


        Parameters
        ----------
        yaxis : string default='raw'
            'raw' - Shows the absolute savings in the native units
            'dollars' - converts to money using self.params.commodityRate
        pointSize : int
            specify the size of the points in the scatter plot

        Returns
        -------
        None - saves plot as class variable

        Raises
        ------
        None

        """

        widthFactor = 1.0
        heightFactor = 1.5

        if yaxis == 'raw':
            ydata = self.postDiff
            ylab = '[' + self.com + ']'

        elif yaxis == 'dollars':
            ydata = self.postDiff * self.params.commodityRate
            ylab = '[$]'

        else:
            raise ValueError('savings_plot requires y-axis to be == '
                             'raw or dollars')
            return

        _color = 'mediumorchid'
        _style = '--'

        # Figure
        fig = plt.figure(figsize=(figW * widthFactor, figH * heightFactor))

        ax0 = plt.subplot2grid((3, 1), (0, 0))

        # Plot 0 - Model post
        ax1 = plt.subplot2grid((3, 1), (0, 0), colspan=1)

        plt.plot(self.post[self.com], label='actual', color='k')
        plt.plot(self.postCalc, label='model', color=_color, linestyle=_style)
        plt.ylabel(self.com)
        plt.title('Post data ' + self.params.paramString)
        plt.legend()

        # plot 1 - savings instant
        ax2 = plt.subplot2grid((3, 1), (1, 0), colspan=1)

        savingsPos = ydata[ydata >= 0]
        savingsNeg = ydata[ydata < 0]

        plt.plot(savingsPos, color='k', linestyle='', marker='.',
                 markersize=pointSize)
        plt.plot(savingsNeg, color='r', linestyle='', marker='.',
                 markersize=pointSize)

#        self.postTest.plot(label='model')
        plt.title('Savings predicted by ' + self.params.paramString)
        plt.ylabel('Savings {}'.format(ylab))
#        plt.legend()

        # Plot 2 - savings cumulative
        ax3 = plt.subplot2grid((3, 1), (2, 0), colspan=1, sharex=ax2)

        cumulative = ydata.cumsum()

        cumPos = cumulative[cumulative >= 0]
        cumNeg = cumulative[cumulative < 0]

        plt.plot(cumPos, color='k', linestyle='', marker='.',
                 markersize=pointSize)
        plt.plot(cumNeg, color='r', linestyle='', marker='.',
                 markersize=pointSize)
        plt.ylabel('Cumulative Savings')
#        plt.legend()

        plt.tight_layout()
        fig.subplots_adjust(hspace=0.2)

        self.savingsPlot = fig


class many_ols():
    """
    Similar to the data holding class, will allow the user to put data into
    an object that will cleanly allow them to run different models and view
    the stats and plots

    """

    """
    My numpydoc description of a kind
    of very exhautive numpydoc format docstring.

    Parameters
    ----------
    first : array_like
        the 1st param name `first`
    second :
        the 2nd param
    third : {'value', 'other'}, optional
        the 3rd param, by default 'value'

    Returns
    -------
    string
        a value in a string

    Raises
    ------
    KeyError
        when a key error
    OtherError
        when an other error
    """

    def __init__(self, pre, post, inputDict):
        self.pre = pre
        self.post = post
        self.com = self.pre.columns[0]
        self.inputDict = inputDict

    def _param_permute(self):
        """
        My numpydoc description of a kind
        of very exhautive numpydoc format docstring.

        Parameters
        ----------
        first : array_like
            the 1st param name `first`
        second :
            the 2nd param
        third : {'value', 'other'}, optional
            the 3rd param, by default 'value'

        Returns
        -------
        string
            a value in a string

        Raises
        ------
        KeyError
            when a key error
        OtherError
            when an other error
        """

        a1 = ['CDH', 'CDH2', '']
        b1 = ['HDH', 'HDH2', '']

        if self.inputDict['paramPermuteList']:
            inputs = self.inputDict['paramPermuteList']
        else:
            inputs = ['', 'C(month)', 'C(weekday)']

        els = []
        for i in range(1, len(inputs)):
            els += [list(x) for x in itertools.combinations(inputs, i)]

        parList = []

        for a in a1:
            for b in b1:
                for e in els:
                    par = [a, b] + e
                    parList.append(" + ".join(filter(None, par)).rstrip(' +'))
        parList.remove('')

        return parList

    def run_all_linear(self):
        """
        My numpydoc description of a kind
        of very exhautive numpydoc format docstring.

        Parameters
        ----------
        first : array_like
            the 1st param name `first`
        second :
            the 2nd param
        third : {'value', 'other'}, optional
            the 3rd param, by default 'value'

        Returns
        -------
        string
            a value in a string

        Raises
        ------
        KeyError
            when a key error
        OtherError
            when an other error
        """

        outputs = {}
        permuteParams = self._param_permute()

        for params in permuteParams:
#            try:
#                self.inputDict['params'] = params
#                outputs[params] = ols_model(self.pre, self.post, self.inputDict)
#            except Exception as e:
#                print('Could not complete model with {}'.format(params))
#                print(e)

            self.inputDict['params'] = params
            outputs[params] = ols_model(self.pre, self.post, self.inputDict)

        self._modelPool = outputs
        self._pool_stats()

    def _pool_stats(self):
        """
        My numpydoc description of a kind
        of very exhautive numpydoc format docstring.

        Parameters
        ----------
        first : array_like
            the 1st param name `first`
        second :
            the 2nd param
        third : {'value', 'other'}, optional
            the 3rd param, by default 'value'

        Returns
        -------
        string
            a value in a string

        Raises
        ------
        KeyError
            when a key error
        OtherError
            when an other error
        """
        try:
            assert(self._modelPool)
        except AssertionError:
            print('You must run "run_all_linear" to generate modelPool before'
                  ' running pool_stats')

        statsPool = {}

        modelNumber = -1

        # Collect stats into dict of dicts
        try:  # py27 and py36 compatibility

            for params, mod in self._modelPool.iteritems():
                modelNumber += 1

                newStatsRow = {'params': params,
                               'AIC': mod.Fit.aic,
                               'R2': mod.Fit.rsquared,
                               'AR2': mod.Fit.rsquared_adj,
                               'cvrmse': mod.Fit.cvrmse,
                               'postDiff': mod.postCumsum,
                               'summary': mod.Fit.summary()}

                statsPool[modelNumber] = newStatsRow

        except AttributeError:  # py27 - dict.iteritems() py36 - dict.items()
            for params, mod in self._modelPool.items():
                modelNumber += 1

                newStatsRow = {'params': params,
                               'AIC': mod.Fit.aic,
                               'R2': mod.Fit.rsquared,
                               'AR2': mod.Fit.rsquared_adj,
                               'cvrmse': mod.Fit.cvrmse,
                               'postDiff': mod.postCumsum,
                               'summary': mod.Fit.summary()}

                statsPool[modelNumber] = newStatsRow

        # Build DataFrame
        self.statsPool = pd.DataFrame(statsPool).T.sort_values('AIC')
        # Re-order columns
        self.statsPool = self.statsPool[['AIC', 'AR2', 'R2', 'cvrmse',
                                         'postDiff', 'params', 'summary']]

    def plot_pool(self, number=5):
        """ Plots the top 5 models after run_all"""
        for i in range(number):
            modParams = self.statsPool['params'].iloc[i]
            tempMod = self._modelPool[modParams]

            tempMod.model_plot2()
            tempMod.stats_plot()
            plt.show()

class remodel():

    """
    My numpydoc description of a kind
    of very exhautive numpydoc format docstring.

    Parameters
    ----------
    first : array_like
        the 1st param name `first`
    second :
        the 2nd param
    third : {'value', 'other'}, optional
        the 3rd param, by default 'value'

    Returns
    -------
    string
        a value in a string

    Raises
    ------
    KeyError
        when a key error
    OtherError
        when an other error
    """
    @staticmethod
    def read_json_params():
        """ Read previous models parameters from the saved json file"""
        # read in Json params locally, split into two dicts
        # return two objects

        jsonName = 'all Params.json' # TODO: Move this to a class varaible
        with open(jsonName) as f:
            data = json.load(f)

        return data
    @staticmethod
    def read_raw_data():
        """ Read previous model's rawData file """
        # find raw data locally, load into pd.dataframe
        # return data

        fileName = 'raw data.xlsx' #TODO: File name will be flexible (right?)
        df = pd.read_excel(fileName, index_col=0,
                           parse_dates=True, infer_datetime_format=True)

        return df
    @staticmethod
    def extend_data_with_pi(data, endDate='y', tags=[]):
        """
        My numpydoc description of a kind
        of very exhautive numpydoc format docstring.

        Parameters
        ----------
        first : array_like
            the 1st param name `first`
        second :
            the 2nd param
        third : {'value', 'other'}, optional
            the 3rd param, by default 'value'

        Returns
        -------
        string
            a value in a string

        Raises
        ------
        KeyError
            when a key error
        OtherError
            when an other error
        """
        # Find last date in data, make that start date for datapull
        startDate = data.index[-1] + pd.Timedelta('1h')
        # Default end date to yesterday

        # Use column headings as tags
        tags = list(data.columns)
        # if custom tags, then pull those and replace columns??? (tricky)

        pi = pi_client()

        newData = pi.get_stream_by_point(tags, start=startDate,
                                         end=endDate, interval='1h')

        # Pull data, concat and return
        combinedData = pd.concat([data, newData], axis=0)

        return combinedData
    @staticmethod
    def DK_II(data, dataParams):
        """
        My numpydoc description of a kind
        of very exhautive numpydoc format docstring.

        Parameters
        ----------
        first : array_like
            the 1st param name `first`
        second :
            the 2nd param
        third : {'value', 'other'}, optional
            the 3rd param, by default 'value'

        Returns
        -------
        string
            a value in a string

        Raises
        ------
        KeyError
            when a key error
        OtherError
            when an other error
        """
        dataParams['dateRanges'][3] = data.index[-1].strftime('%Y-%m-%d')
        dataParams['IQRmult'] = 6

#        return dataParams
        dk = data_keeper(data, dataParams)

        dk.default_clean()

        # make DK
        # Default Clean
        # Defualt plots
        # return DK
        return dk

    @staticmethod
    def MC_II(pre, post, modelParams):
        """
        My numpydoc description of a kind
        of very exhautive numpydoc format docstring.

        Parameters
        ----------
        first : array_like
            the 1st param name `first`
        second :
            the 2nd param
        third : {'value', 'other'}, optional
            the 3rd param, by default 'value'

        Returns
        -------
        string
            a value in a string

        Raises
        ------
        KeyError
            when a key error
        OtherError
            when an other error
        """

        mc = ols_model(pre, post, modelParams)

        mc.calculate_kfold()
        mc.calculate_vif()

        return mc


# =============================================================================
# Functions
# =============================================================================

def _create_new_save_directory(dirName=None, useDate=False, _index=None):
    """Create new directory where the name and date and be specified
    New directories are created recursively is there is a name match
    """

    """
    My numpydoc description of a kind
    of very exhautive numpydoc format docstring.

    Parameters
    ----------
    first : array_like
        the 1st param name `first`
    second :
        the 2nd param
    third : {'value', 'other'}, optional
        the 3rd param, by default 'value'

    Returns
    -------
    string
        a value in a string

    Raises
    ------
    KeyError
        when a key error
    OtherError
        when an other error
    """

    currentTime = datetime.now().strftime('%Y-%m-%d_%H%M%S')

    if dirName:
        if useDate:
            tryDir = "{}_{}".format(dirName, currentTime)
        else:
            tryDir = dirName  # pass
    else:  # no dir name just becomes the date
        tryDir = currentTime

    try:
        if _index:
            mkdir("{} ({})".format(tryDir, _index))
        else:
            mkdir(tryDir)

    except FileExistsError:
        print('Warning: Folder {} ({}) already exists.'.format(tryDir, _index))

        if _index:
            _index += 1
        else:
            _index = 1

        _create_new_save_directory(dirName=dirName, useDate=useDate,
                                   _index=_index)

    return tryDir


def _prove_completion(mc):
    """ Checks if certain items exist in model object 'mc'"""

    completed = {}

    try:
        assert(isinstance(mc.vif, pd.DataFrame))
        completed['VIF'] = True
    except AttributeError:
        print('create_archive warning: VIF has not been run')
        completed['VIF'] = False
    try:
        assert(isinstance(mc.kfoldStats, pd.DataFrame))
        completed['kfold'] = True
    except AttributeError:
        print('create_archive warning: kfold has not been run')
        completed['VIF'] = False

    return completed


def create_archive(dk, mc,
                   saveFigs=False,
                   customDirectoryName=None):

    """
    Saves many files to archive a data/model run.

    These items are used for compiling the M&V report
    and re-running models using the __Model__ notebook

    Parameters
    ----------
    dk : instance of data_keeper
        the 1st param name `first`
    second :
        the 2nd param
    third : {'value', 'other'}, optional
        the 3rd param, by default 'value'

    Returns
    -------
    string
        a value in a string

    Raises
    ------
    KeyError
        when a key error
    OtherError
        when an other error
    """

    """
    Make a function that tests if the model cross validates well using
    the k fold split of random data.

    Maybe this is the wrong class for this function?

    """

    saveDirectory = _create_new_save_directory(dirName=dk.com, useDate=True)

    # Check to see if VIF kfold and other functions are completed
    # if not completed then they are ignored from the excel writer
    completionDict = _prove_completion(mc)

    # Write excel file
    excelSaveString = '{}/{} results.xlsx'.format(saveDirectory,
                                                  mc.com)

    writer = pd.ExcelWriter(excelSaveString, engine='xlsxwriter')

    if completionDict['VIF']:
        mc.vif.to_excel(writer, sheet_name='VIF')

    if completionDict['kfold']:

        concatedKfold = pd.concat([mc.kfoldStats, mc.kfoldRelative], axis=1)
        concatedKfold.to_excel(writer, sheet_name='Kfold')

    # write ols_model.Fit().summary() in sections
    # dfList is 3 seperate stats summaries, so we need to write them to the
    # same sheet iteratively and increment the startrow by the shape + 1
    dfList = pd.read_html(mc.Fit.summary().as_html())
    startRow = 0
    for df in dfList:
        df.to_excel(writer, sheet_name='Summary',
                    startrow=startRow, startcol=0,
                    header=False, index=False)
        startRow += df.shape[0] + 1

    # Write all params into a json for reading .
    # Necessary to access __dict_ since it contains the default/hidden params
    allParams = {'dataParams': dk.params.__dict__,
                 'modelParams': mc.params.__dict__}
    with open('{}/all Params.json'.format(saveDirectory), 'w') as f:
        json.dump(allParams, f, indent=4)

    # TODO: Add pickling of raw data?
    dk.rawData.to_excel('{}/raw data.xlsx'.format(saveDirectory))

    # Output images
    if saveFigs:
        mc.mod1Plot.savefig('{}/Model plot.png'.format(saveDirectory))
        mc.savingsPlot.savefig('{}/Savings plot.png'.format(saveDirectory))
