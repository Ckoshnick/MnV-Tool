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
        dk.rawData - for loading into reModel notebook

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

---- MOVING TO GITHUB ALL FUTURE CHANGES WILL BE RECORDED THERE ----


known issues:
    - TODO: Fix doc strings containing "exhautive numpydoc format docstring."
    - Models are limited to "D" or "H" resampled due to the implementation in
        how OAT is turned into degree days (eg can not do a 3hr long interval)


Last update 8/2/18 - koshnick
"""

# =============================================================================
# --- Imports
# =============================================================================

# import sys
import json
import itertools
import scipy.stats

import numpy as np
import pandas as pd
import seaborn as sns

import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from os import path, mkdir
from math import sqrt, floor
from shutil import copyfile
from datetime import datetime
from PI_Client import v2_1

from sklearn.model_selection import KFold
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

# =============================================================================
# --- Constants
# =============================================================================

pathPrefix = path.dirname(path.abspath(__file__))
plt.rcParams.update({'figure.max_open_warning': 0})
sns.set()  # Enable to turn on seaborn plotting
pi = v2_1.pi_client()
piOATpoint = 'aiTIT4045'

figW = 18
figH = 6

version = 'Version 1.6.2'

# =============================================================================
# --- Classes
# =============================================================================


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
         'floor': -1,
         'ceiling': 10000,

         'resampleRate' : 'D',
         'sliceType' : 'half', #half, middate, ranges
         'midDate' : None, #only needed with sliceType : 'middate'
         'dateRanges' : None #required for 'ranges'
         'OATsource' : 'file',
         'OATname' : None}

    column : int or str (0)
        if str - name of column to select if int - column index
    IQRmult : float (3.0)
        Number that the IQR will be bultiplied by to adjust floor/ceiling
        see self.remove_outliers() function
    IQR : str ('y')
        if IQR == 'y' then the outlier detection will use IQR method
        see self.remove_outliers() function
    floor : int (-1)
        Lowest value any raw data point should take on
        see self.remove_outliers() function
    ceiling : int (10000)
        Maximum value any raw data point should take on
        see self.remove_outliers() function
    resampleRate : 'intstr ('D')
        The resampling rate that will be impressed upon the data before passing
        the data off to the modeling class
    sliceType : str ('half')
        options are -half, middate, ranges- see self.data_slice() function
    midDate : str (None)
        'yyyy-mm-dd' formatted date, used for sliceType = 'middate'
    'dateRanges' : list (None)
        ['yyyy-mm-dd'] x4 dates used for sliceType = 'ranges'
        see self.data_slice() function for details
    'OATsource' : str ('file')
        Defines the source of the OAT data for calculating HDH/CDH
        see self.add_degree_hours() for details
    'OATname' : str
        The name of the OAT column for calculating HDH/CDH when the OAT is
        supplied along with the data. This option is only needed if OATsource =
        'self'
        see self.add_degree_hours() for details

    Returns
    -------
    class instance
        instance.attibutes

    Raises
    ------
    None

    """

    def __init__(self,
                 column=0,
                 IQRmult=3,
                 IQR='y',
                 modifiedInterval=None,
                 floor=-1,
                 ceiling=10000,
                 resampleRate='D',
                 sliceType='half',
                 midDate=None,
                 dateRanges=None,
                 OATsource='pi',
                 OATname=None):

        self.column = column
        self.modifiedInterval = None  # Explicitly None, will be calculated
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
        dk.raw - raw data DataFrame
        dk.modifiedData - working data DataFrame
        dk.restoreData - a copy made before modifiedData is changed # disabled
        dk.pre - pre period DataFrame created by data_slice()
        dk.post - post period DataFrame created by  data_slice()

    """

    # Data Class init

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

        # modifiedData is the working variable for all data modifications
        self.modifiedData = self.rawData[self.com].to_frame()

        self.params.modifiedInterval = (
                _interval_checker(self.modifiedData, silence=False,
                                     symbolOutput=True))

        # Use the OAT loaded into instance if present, or PI data if selected
        # otherwise OAT data will be loaded from OAT master
        if self.params.OATsource == 'self':
            self.OAT = data[self.params.OATname]
        elif self.params.OATsource == 'pi':
            self.OAT = (
                pi.get_stream_by_point(piOATpoint,
                                       start=self.modifiedData.index[0],
                                       end=self.modifiedData.index[-1],
                                       interval='1h'))
            self.params.OATname = piOATpoint

    def undo(self):
        """ Allows the user to go back one step in data modification """
        # TODO: Retired - remove
        self.modifiedData = self.restoreData.copy()

# =============================================================================
#  CLEANING
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

    def remove_outliers(self, floor=None, ceiling=None, IQR=None):
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
        if not floor:
            floor = self.params.floor
        if not ceiling:
            ceiling = self.params.ceiling
        if not IQR:
            IQR = self.params.IQR

        # Store copy incase self.undo() invoked
        # self.restoreData = self.modifiedData.copy() #disabled july 31,18 v1.4

        # temp needed for being temporary modifiedData
        temp = self.modifiedData.copy()

        if IQR:
            IQRupper, IQRlower = self._IQR()
            print('IQRupper', IQRupper, ';', 'IQRlower', IQRlower)

            if IQRupper < ceiling:
                ceiling = IQRupper
                print('Ceiling adjusted by IQR - Now {0:.2f}'.format(ceiling))
            if IQRlower > floor:
                floor = IQRlower
                print('Floor adjusted by IQR   - Now {0:.2f}'.format(floor))

        # Select data where floor < data < ceiling
        if floor is not None:
            temp = temp.where(temp > floor, other=np.nan)
        if ceiling is not None:
            temp = temp.where(temp < ceiling, other=np.nan)
        else:
            pass

        modIndex = self.modifiedData.index
        indexDifference = modIndex.difference(temp.dropna().index)

        self.outliers = self.modifiedData[self.com][indexDifference]
        self.modifiedData[self.com] = temp
        print('')

    def _resample(self, resampleRate=None, aggFun='mean'):
        """ Calls resample rate pandas native with default argument handling"""

        # self.restoreData = self.modifiedData.copy() #disabled july 31,18 v1.4

        if resampleRate is None:
            resampleRate = self.params.resampleRate

        # Need this keep index hack becuase of how sum and mean differ
        # ie. [nan, nan].sum() = 0 and [nan nan].mean() = nan. KeepIndex helps
        # to make sure days of missing data dont show up as zeros when they
        # need to be blank (nan) instead.
        keepIndex = self.modifiedData.resample(resampleRate).mean().dropna()

        self.modifiedData = (
                self.modifiedData
                .resample(resampleRate)
                .agg(aggFun))

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

        if self.params.OATsource == 'file':
            hours = _calculate_degree_hours(oatData=None,
                                            by=self.params.modifiedInterval,
                                            cutoff=cutoff)

        elif self.params.OATsource == 'self':
            hours = _calculate_degree_hours(oatData=self.OAT,
                                            by=self.params.modifiedInterval,
                                            cutoff=cutoff)

        # HACK: We're essentially turning the PI data in self, but not treating
        # it that way with the following line. Refactor the 'self' handling
        # to make this cleaner. Maybe if 'self' is used in params, but
        # there is no actual oAT (how to check that ?? x.x) then we switch to
        # pulling from aiTIT. or maybe it needs to be implemented like this..
        elif self.params.OATsource == 'pi':
            hours = _calculate_degree_hours(oatData=self.OAT,
                                            by=self.params.modifiedInterval,
                                            cutoff=cutoff)

        self.modifiedData['HDH'] = hours['HDH']
        self.modifiedData['CDH'] = hours['CDH']

        self.modifiedData['HDH2'] = self.modifiedData['HDH'] ** 2
        self.modifiedData['CDH2'] = self.modifiedData['CDH'] ** 2

    def add_time_columns(self, daytime=(8, 20)):
        """ Calls the mypy function build_time_columns rebranded in this file
        """
        self.modifiedData = _build_time_columns(self.modifiedData,
                                                daytime=daytime)

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
            midPoint = int(floor(dataLength) / 2)

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
                ranges = self.params.dateRanges
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
        # HACK: Remove this since keep index works? needs testing
        self.modifiedData = self.modifiedData.dropna()

        self.add_degree_hours()
        self._resample(aggFun='sum')
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
        plt.subplot2grid((1, 5), (0, 0))

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
        plt.subplot2grid((1, 5), (0, 1), colspan=4)

        plt.plot(noOutliers, color='k', linestyle='', marker='.')
        plt.plot(self.outliers, color='r', linestyle='', marker='.')

        plt.title('Outlier removal result. interval = raw'.format(
            str(self.params.modifiedDataInterval)))

        if yrange:
            plt.ylim(yrange)
        plt.show()

        self.outlierPlot = fig

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
        plt.subplot2grid((1, 5), (0, 0))

        # Box plot
        ax1 = plt.subplot2grid((1, 5), (0, 0), colspan=1)

        sns.boxplot(data=self.modifiedData[self.com], ax=ax1)
        plt.ylabel(self.com)
        plt.title('clean & resampled data')

        # Scatter plot
        plt.subplot2grid((1, 5), (0, 1), colspan=4)

        plt.plot(self.modifiedData[self.com],
                 color='k', linestyle='', marker='.')

        plt.title('Outlier and resample removal result. interval = {}'.format(
            str(self.params.modifiedDataInterval)))

        if yrange:
            plt.ylim(yrange)

        self._resampledPlot = fig
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

    var : str or list (['month', 'weekday'])
        All of the params to run the ols model with
        if list or str the data is converted in _convert_var to populate
        self.varList and self.varString
    varList : list of strings (None)
        All variables that will be ran in many_ols.test_all_linear()
        populated by _convert_var
    varString : str (None)
        All variables that will be ran in many_ols.test_all_linear()
        populated by _convert_var
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
    varPermuteList : list of strings
        List of strings of all variables to be permuted and considered for the
        many_ols class


    Returns
    -------
    class instance
        instance.attibutes

    # TODO: Think of a good name for model class. OM? MC? mod?


    Raises
    ------
    ValueError
        When var is not a list of strings or a string
    """

    def __init__(self,
                 var=['month', 'weekday'],
                 varList=None,
                 varString=None,
                 testTrainSplit='random',
                 randomState=4291990,
                 testSize=0.2,
                 commodityRate=0.056,
                 varPermuteList=['', 'C(month)', 'C(weekday)'],
                 defaultPermutes=None):

        self.var = var
        self._convert_var(var)

        self.testTrainSplit = testTrainSplit
        self.randomState = randomState
        self.testSize = testSize
        self.commodityRate = commodityRate
        self.varPermuteList = varPermuteList
        self.defaultPermutes = defaultPermutes

    def _convert_var(self, var):
        """
        Convert the var from the list form ['month','hour']
        to the string form 'C(month) + C(hour)' or vice versa. Important for
        allowing flexibility in how the vars are input and it helps the build
        dummy section play nice.

        Parameters
        ----------
        var : string or list
            All of the variables to run the ols model with

        Returns
        -------
        list or string
            if list input, string returns. if string input list returns

        Raises
        ------
        ValueError
            Input var must be list or string
        """

        DD = ['HDH', 'CDH', 'CDH2', 'HDH2']

        # convert to list, varString
        if isinstance(var, list):
            self.varList = list(var)
            newVarList = []
            for variable in self.varList:
                if variable in DD:
                    newVarList.append(variable)
                else:
                    newVarList.append("C({})".format(variable))

            self.varString = ' + '.join(newVarList)

        # conver to string, varList
        elif isinstance(var, str):
            varList = var.split('+')
            varList = [x.replace("C(", "").replace(")", "").strip(' ')
                       for x in varList]
            self.varList = varList
            self.varString = var

        else:
            raise ValueError('var input must be either string or list as '
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

    Other functionality of this class it to provide statistics and tests to
    ensure that the model is appropriate for the data being described. These
    test are to calculated the Variance Inflation Factor (VIF) calculted by
    "calcuate_vif()". Another test is the k-fold.

    Lastly this model also predicts savings for the post period using TMY data
    as an input variable. These outputs can all be stored in an 'archive' by
    the mnv.create_archive() method.

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
        pre - The pre period data from dk
        post - The post period data from dk
        Model - The sm.OLS model generated using the input vars
        Fit - The sm.OLS.Fit object that calculates predictions
        VIF - Variance Inflation Factor Results
        kfoldStats - KFold MSE and values normalized to the mean MSE
        postCalc - Predicted energy use for the post period
        postDiffSum - The cumulative sum for the differnce of model and actual
        tmyFYESum - Calculated difference from postEnd(YTD) to FYEnd
        tmyYearSum - Calculated difference from postEnd(YTD) to YTD + 1year
        data - pd.df container of most predictions for easy calculations
        F - Fraction of energy saved in post period
        uncertainty - uncertainty in the fraction of savings F

    """

    def __init__(self, pre, post, inputDict):

        self.params = ModelParameters(**inputDict)

        self.pre = pre
        self.post = post
        self.com = pre.columns[0]

        # Important Date objects
        self.preStart = pre.index[0]
        self.preEnd = pre.index[-1]
        self.postStart = post.index[0]
        self.postEnd = post.index[-1]  # Same as postEnd
        self.oneYearEnd = self.postStart + pd.offsets.DateOffset(years=1)

        self.dataInterval = _interval_checker(pre, silence=True,
                                              symbolOutput=True)

        # Begin modeling
        self.split_test_train(how=self.params.testTrainSplit)
        self.Model = smf.ols(self.com + '~' + self.params.varString,
                             data=self.train)
        self.Fit = self.Model.fit()

        # TODO Is this math correct?
        self.Fit.cvrmse = (sqrt(self.Fit.mse_resid)
                           / self.train[self.com].mean())

        # Make predictions called "Calcs"  # Keep these through the purge
        self.trainCalc = self.Fit.predict(self.train)
        self.testCalc = self.Fit.predict(self.test)
        self.postCalc = self.Fit.predict(post)

        # Needed for stats_plot
        self.trainDiff = self.trainCalc - self.train[self.com]

        # Aggregate commodity data and calculations into "data" df
        self.data = pre[self.com].to_frame()
        self.data.columns = ['pre']

        # Need to use custom join to extend axis
        self._custom_join(self.post[self.com], newName='post')

        # Once axis is extended, assigning values to columns works fine
        self.data['preModel'] = pd.concat(
                [self.trainCalc, self.testCalc, self.postCalc], axis=0
                ).sort_index()

        self._calculate_diff_and_sum(
                self.data['preModel'][self.postStart:self.postEnd],
                self.data['post'][self.postStart:self.postEnd],
                assignment='post')

        # Variables to be filled when needed, all checked if exist first
        self.postModel = None
        self.tmy = None

    def _custom_join(self, newCol, newName=None):
        """Appends a column to the df "data" and allows for column name
        changes and makes sure to extend the timeseries axis with an outer join
        """

        if isinstance(newCol, pd.Series):
            newCol = newCol.to_frame()
        if newName:
            newCol.columns = [newName]

        if newName in self.data.columns:
            print('Warning: {} already in self.data\n'
                  'Unable to _custom_join()'.format(newName))
        else:
            # Reset index, join re-set index to avoid index overlap error
            self.data = self.data.join(newCol, how='outer')

#               TODO: This below comment should fix the overlap problem
#            self.data = self.data.reset_index().join(newCol.reset_index(),
#                                         on='index', how='outer')
#            self.data = self.data.drop_duplicates(cols='index')
#            self.data.set_index('index', inplace=True)

    def _find_FYE(self, date, end=True):
        """ Finds next date of the 4/1/yyyy FYE based on input date"""

        year = int(date.year)
        month = int(date.month)

        if month >= 4:
            year += 1
        if end:
            pass
        else:
            year -= 1

        endDate = pd.to_datetime("{}-04-01".format(year))

        return endDate

    def _remove_degree_days(self, var):
        """This removes and reinstalls the degree days for dummy creation"""

        var = list(var)  # Ghetto copy
        dumList = ['CDH', 'HDH', 'CDH2', 'HDH2']
        removed = []

        for dum in dumList:
            if dum in var:
                var.remove(dum)
                removed.append(dum)
            else:
                pass

        return var, removed

    def _calculate_diff_and_sum(self, expected, actual, assignment=None):
        """ A method to calculate the diff columns and the cumsum of the diff
        column inspired by the following (retired) statement.

        self.data['postDiff'] = (self.data['preModel'][self.postStart:self.postEnd]
                                 - self.data['post'])
        """

        diff = expected - actual

        # XXX: Remove this function once new savings suite complete (?)
        # Hack: .dropna() fixes issue where OAT (and thus HDD) is missing
        if assignment == 'tmy':
            self._custom_join(diff, newName='tmyDiff')
            self.tmyFYESum = diff[self.postEnd:self.FYEnd].cumsum()[-1]
            self.tmyYearSum = diff[self.FYEnd:self.oneYearEnd].cumsum()[-1]
        elif assignment == 'post':
            cumsum = diff.cumsum().dropna()[-1]
            self._custom_join(diff, newName='postDiff')
            self.postDiffSum = cumsum


    def _generate_savings_intervals(self):
        """ Using the postStart and postEnd dates, determine where the FY and
        annual date intervals fall"""

        PE = self.postEnd
        PS = self.postStart
#        print(PS, PE)
        oneYear = pd.offsets.DateOffset(years=1)

        # find last FY dates
        FY11 = self._find_FYE(PE - oneYear, end=False)
        FY12 = FY11 + oneYear
        # Find this FY dates
        FY21 = FY12  # + a step?
        FY22 = FY12 + oneYear

#        print(FY11, FY12, FY21, FY22)

        # Find Last year's dates
        yearDiff = PE.year - self.postStart.year
        PSvirt = PE - pd.offsets.DateOffset(years=yearDiff)

        if PSvirt < PS:
            offset = -1
        if PSvirt >= PS:
            offset = 0

        A11 = PS + pd.offsets.DateOffset(years=(yearDiff + offset - 1))
        A12 = A11 + oneYear
        # Find this year's dates
        A21 = A12  # + a step?
        A22 = A12 + oneYear

#        print(A11, A12, A21, A22)
        validation = dict(lastFY=PS < FY12,
                          lastYear=PE > (PS + oneYear))

        if PS > FY11 and validation['lastFY']:
            FY11 = PS
        else:
            FY11 = None
            FY12 = None

        if PS > FY21:
            FY21 = PS

        if not validation['lastYear']:
            A11 = None
            A12 = None

        index = ['Last FY', 'This FY', 'Last Year', 'This Year', 'Total']
        cols = ['Actual', 'TMY prediction', 'Combined', 'Start', 'Middle', 'End']

        df = pd.DataFrame(columns=cols, index=index)
        df['Start'] = pd.Series(data=[FY11, FY21, A11, A21, PS], index=index)
        df['Middle'] = pd.Series(data=['', PE, '', PE, PE], index=index)
        df['End'] = pd.Series(data=[FY12, FY22, A12, A22, PE], index=index)

        self.savingsSummary = df

    def generate_savings_summary(self):
        """ Compile the actual and or tmy savings values """

        self.calculate_tmy_models()
        self._generate_savings_intervals()
        PE = self.postEnd

        for index, row in self.savingsSummary.iterrows():
#            print(row)
            if not row['Start']:
                continue

            if PE > row['Start'] and PE <= row['End']:
                row['Actual'] = self.data['postDiff'][row['Start']:PE].cumsum()[-1]
                row['TMY prediction'] = self.data['tmyDiff'][PE:row['End']].cumsum()[-1]
            elif PE > row['End']:
                row['Actual'] = self.data['postDiff'][row['Start']:row['End']].cumsum()[-1]
                row['TMY prediction'] = row['TMY prediction']

            self.savingsSummary.loc[index] = row

        self.savingsSummary[['Actual', 'TMY prediction']] *= self.params.commodityRate

        self.savingsSummary['Combined'] = self.savingsSummary[['Actual','TMY prediction']].sum(axis=1).round(0)

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

        # TODO: IF VIF HAS TOO FEW COLUMNS THEN ABORT
        # Get dummies
        dummyColumns, removed = self._remove_degree_days(self.params.varList)

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
        testdf = add_constant(testdf)

        # Get VIF values
        vif_values = [variance_inflation_factor(testdf.values, i)
                      for i in range(testdf.shape[1])]
        # Format into nice pd.DataFrame
        vif = pd.DataFrame(index=testdf.columns,
                           data=vif_values,
                           columns=['VIF'])

        self.vif = vif

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
        2. Fit model var to all folds
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
            results of KFold validation is stored in self.kfoldStats

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

            # in this function to maintaine consitency
            Fit = smf.ols(self.com + '~' + self.params.varString, train).fit()

            newStatsRow = {'AR2': Fit.rsquared_adj,
                           'mse': Fit.mse_resid,
#                           'postDiff': Fit.postCumsum,
                           }

            statsPool[foldNumber] = newStatsRow

        # Build DataFrame
        kfoldStats = pd.DataFrame(statsPool).T.sort_values('mse')
        # Re-order columns
        kfoldStats = kfoldStats[['AR2', 'mse']]

        # Building relative MSE values
        mseMean = round(kfoldStats['mse'].mean())
        kfoldRelative = kfoldStats['mse'] / mseMean * 100

        kfoldStats['pct. of mean'] = kfoldRelative

        self.kfoldStats = kfoldStats

    def calculate_F_uncertainty(self, confidence=90):
        """
        Following ASHRAE Guideline 14-2002
        "Measurement of Energy and Demand Savings"

        U = [t * 1.26 * CVRMSE * sqrt(n + 2)] / [F * sqrt(n * m)]

        t = t-statistic (calculated using scipy.stats.t.ppf())
        CVRMSE = Normalized Root mean squared Error
        n = number of data points or periods in the baseline period
        F = approximate percentage of baseline energy that is saved. This
            percentage should be derived from the m periods of the reporting
            period. Before savings are actually achieved, the predicted savings
            may be used in computed F for purposed of designing the savings
            determination algorithm.
        m = number of periods in the post-retrofit savings reporting period

        U = relative uncertainty in a reported energy saving, expressed as
            a percentage of the savings

        """
        self.F = (self.postDiffSum
                  / self.data['preModel'][self.postStart:self.postEnd].cumsum()[-1])

        t = scipy.stats.t.ppf(confidence/100, len(self.post))

        numer = t * 1.26 * self.Fit.cvrmse * sqrt(len(self.train) + 2)
        denom = self.F * sqrt(len(self.train) * len(self.post))

        self.uncertainty = numer / denom

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

        # load in params if not given
        if testSize is None:
            testSize = self.params.testSize
        if randomState is None:
            randomState = self.params.randomState

        how = how.lower()
        length = len(self.pre)

        if how == 'simple':
            # Must take 1-testsize, since testsize is the small fraction
            splitIndex = int(round(length * (1 - testSize)))

            self.train = self.pre.iloc[0:splitIndex]
            self.test = self.pre.iloc[splitIndex:length]

        elif how == 'random':

            self._get_folds(randomState)

            self.train = self.pre.iloc[self._folds[0][0]]
            self.test = self.pre.iloc[self._folds[0][1]]

    def _get_folds(self, randomState):
        """ Splits the index into 1/(testSize) folds, stores in self._folds"""
        # Calculate # of folds from 1/testSize
        folds = int(round(1 / self.params.testSize))

        # Use the scikitlearn kfolds function to generate randomized folds
        kf = KFold(n_splits=folds,
                   shuffle=True,
                   random_state=randomState)

        self._folds = [x for x in kf.split(self.pre)]

    def _get_tmy_data(self):
        """ Pull the TMY data from the PI tag 'Future_TMY' and store it to be
        used for calculating the TMY calcs

        Parameters
        ----------
        starteDate : str ('pre')
            'yyyy-mm-dd' formatted string
            if 'pre', use the timeStamp of the begining of the 'pre' data set
        endDate : str ('fiscal')
            'yyyy' formatted string or 'yyyy-mm-dd' string
            if 'fiscal' use the last day of this fiscal year
            if 'yyyy' do the same as 'year' but with specified year

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

        startDate = self.preStart
        endDate = self.oneYearEnd

        tmy = pi.get_stream_by_point(['Future_TMY'], start=startDate,
                                     end=endDate, interval='1h')

        hours = _calculate_degree_hours(tmy, by=self.dataInterval)

#        hours['HDH2'] = hours['HDH'] ** 2
#        hours['CDH2'] = hours['CDH'] ** 2

        # TODO: rename TMY to TMYDH? or something that explains there is no OAT, or program the OAT back in? Do we really want the OAT since it is going to be mean() or sum()?
        self.tmy = hours

    def _make_tmy_pre(self):
        """ Predict building behavior using tmy data """

        tmyPre = self.Fit.predict(self.tmy).to_frame()
        tmyPre.columns = ['tmyPre']

        self._custom_join(tmyPre)

    def _make_tmy_post(self):
        """ Predict building behavior using tmy data and post data"""

        tmyPost = self.postModel.fit().predict(self.tmy).to_frame()
        tmyPost.columns = ['tmyPost']

        self._custom_join(tmyPost)

    def calculate_tmy_models(self):
        """ This function calculates the tmy pre and post models, then finds
        the difference between them

        Parameters
        ----------
        startDate : str
            see _get_tmy_data
        endDate : str
            see _get_tmy_data

        Returns
        -------
        None
            stores self.tmy

        Raises
        ------
        None

        """

        if self.postModel is None:
            minusMonth = self.params.varString.replace('+ C(month)', '')
            self.params.tmyVarString = minusMonth

            self.postModel = smf.ols(self.com + '~' + minusMonth,
                                     data=self.post)

        if self.tmy is None:  # Set self.tmy = None to reset lockout
            self._get_tmy_data()
            self.tmy = _build_time_columns(self.tmy)

            self._make_tmy_post()
            self._make_tmy_pre()

        self.data['tmyDiff'] = - self.data['tmyPost'] + self.data['tmyPre']

    def plot_tmy_comparison(self):
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

        _color = 'tomato'
        _style = '--'

        widthFactor = 1.0
        heightFactor = 2.0

        fig = plt.figure(figsize=(figW * widthFactor, figH * heightFactor))
        plt.subplot2grid((3, 1), (0, 0))

        # Plot 1
        plt.subplot2grid((3, 1), (0, 0), colspan=1)

        plt.plot(self.data['pre'], label='actual', color='k')
        plt.plot(self.data['tmyPre'][self.preStart:self.preEnd],
                 label='tmy_pre_model', color=_color, linestyle=_style)

        plt.title('tmy Pre ' + self.params.varString)
        plt.ylabel(self.com)
        plt.legend()

        # Plot 2
        plt.subplot2grid((3, 1), (1, 0), colspan=1)

        plt.plot(self.data['post'], label='actual', color='k')
        plt.plot(self.data['tmyPost'][self.postStart:self.postEnd],
                 label='tmy_Post_model', color=_color, linestyle=_style)

        plt.ylabel(self.com)
        plt.title('tmy Post ' + self.params.tmyVarString)
        plt.legend()
        plt.tight_layout()

        # Plot 3
        plt.subplot2grid((3, 1), (2, 0), colspan=1)

        plt.plot(self.data['tmyPre'][self.postStart:self.oneYearEnd],
                 label='tmy_Pre_model', color='darkgreen')
        plt.plot(self.data['tmyPost'][self.postStart:self.oneYearEnd],
                 label='tmy_Post_', color='deepskyblue')

        plt.ylabel(self.com)
        plt.title('TMY comparison')
        plt.legend()
        plt.tight_layout()

        self.tmyPlot = fig

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
        plt.subplot2grid((2, 1), (0, 0))

        plt.subplot2grid((2, 1), (0, 0), colspan=1)

        plt.plot(self.test[self.com], label='actual', color='k')
        plt.plot(self.testCalc, label='model', color=_color, linestyle=_style)

        plt.title('Test data ' + self.params.varString)
        plt.ylabel(self.com)
        plt.legend()

        plt.subplot2grid((2, 1), (1, 0), colspan=1)

        plt.plot(self.post[self.com], label='actual', color='k')
        plt.plot(self.postCalc, label='model', color=_color, linestyle=_style)

        plt.ylabel(self.com)
        plt.title('Post data ' + self.params.varString)
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
        plt.subplot2grid((1, 1), (0, 0))

        plt.subplot2grid((1, 1), (0, 0), colspan=1)

        plt.plot(self.test[self.com], label='actual', color='k')
        plt.plot(self.testCalc, label='model', color=_color, linestyle=_style)

        plt.title('Test data ' + self.params.varString)
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
        plt.subplot2grid((1, 3), (0, 0))

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

    def savings_plot(self, yaxis='raw', pointSize=6):
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
            ydata = self.data['postDiff']
            ylab = '[' + self.com + ']'

        elif yaxis == 'dollars':
            ydata = self.data['postDiff'] * self.params.commodityRate
            ylab = '[$]'

        else:
            raise ValueError('savings_plot requires y-axis to be == '
                             'raw or dollars')
            return

        _color = 'mediumorchid'
        _style = '--'

        # Figure
        fig = plt.figure(figsize=(figW * widthFactor, figH * heightFactor))

        plt.subplot2grid((3, 1), (0, 0))

        # Plot 0 - Model post
        plt.subplot2grid((3, 1), (0, 0), colspan=1)

        plt.plot(self.post[self.com], label='actual', color='k')
        plt.plot(self.postCalc, label='model', color=_color, linestyle=_style)
        plt.ylabel(self.com)
        plt.title('Post data ' + self.params.varString)
        plt.legend()

        # plot 1 - savings instant
        ax2 = plt.subplot2grid((3, 1), (1, 0), colspan=1)

        savingsPos = ydata[ydata >= 0]
        savingsNeg = ydata[ydata < 0]

        plt.plot(savingsPos, color='#76cd26', linestyle='', marker='.',
                 markersize=pointSize)
        plt.plot(savingsNeg, color='r', linestyle='', marker='.',
                 markersize=pointSize)

#        self.postTest.plot(label='model')
        plt.title('Savings predicted by ' + self.params.varString)
        plt.ylabel('Difference\n{}'.format(ylab))
#        plt.legend()

        # Plot 2 - savings cumulative
        plt.subplot2grid((3, 1), (2, 0), colspan=1, sharex=ax2)

        cumulative = ydata.cumsum()

        cumPos = cumulative[cumulative >= 0]
        cumNeg = cumulative[cumulative < 0]

        plt.plot(cumPos, color='#76cd26', linestyle='', marker='.',
                 markersize=pointSize)
        plt.plot(cumNeg, color='r', linestyle='', marker='.',
                 markersize=pointSize)
        plt.ylabel('Cumulative\n{}'.format(ylab))
#        plt.legend()

        plt.tight_layout()
        fig.subplots_adjust(hspace=0.2)

        self.savingsPlot = fig
        plt.show()

class many_ols():
    """
    Similar to the data holding class, will allow the user to put data into
    an object that will cleanly allow them to run different models and view
    the stats and plots

    Parameters
    ----------
    pre : pd.DataFrame
        df of the pre-period data with HDH and time column variables
    post : pd.DataFrame
        df of the post-period data with HDH and time column variables
    inputDict : dict
        **kwargs type dict to be passed into the ModelParameters class

    Returns
    -------
    class instance: mod

        notable attributes
        ------------------
        mod.postCalc
        mod.postDiffCumsum
        # TODO: Finalize

    """

    def __init__(self, pre, post, inputDict):
        self.pre = pre
        self.post = post
        self.com = self.pre.columns[0]
        self.inputDict = inputDict

        if 'defaultPermutes' in list(inputDict.keys()):
            self.defaultPermutes = inputDict['defaultPermutes']
        else:
            self.defaultPermutes = [['CDH', 'CDH2', ''], ['HDH', 'HDH2', '']]


    #TODO: Finish implementing the default permutes here so that I can pass
    #  in more model vars to this thing

    def _var_permute(self):
        """ Creates all permutations of HDH or HDH2, CDH or CDH2, and the vars
        passed in to 'varPermuteList including ""
        """

        if self.defaultPermutes:
            permuteGroup = self.defaultPermutes
        else:
            permuteGroup = []

        if self.inputDict['varPermuteList']:
            permutes = self.inputDict['varPermuteList']
        else:
            permutes = ['', 'C(month)', 'C(weekday)']


#        for i in range(1, len(inputs)):
        #els += [list(x) for x in itertools.combinations(inputs, i)]

        permutesList = []
        for i in range(1, len(permutes)):
            permutesList += [list(x) for x in itertools.combinations(permutes, i)]

        for i, item in enumerate(permutesList):
            permutesList[i] = " + ".join(item).strip(' +')

        permuteGroup.append(permutesList)

        permutedList = []
        for item in list(itertools.product(*permuteGroup)):
            permutedList.append(" + ".join(filter(None, item)).rstrip(' +'))
        # Remove the var combo where all options are ""

#        print(permutedList)
        try:
            permutedList.remove('')
        except ValueError:
            pass

        return permutedList

    def run_all_linear(self):
        """Takes in every combination of vars generated from _var_permute and
        creates a 'pool' of linear models
        # TODO: this function can be slow, maybe trim the OLS init as much as
        possible to increase speed here

        """

        print('Entering run_all_linear()...\n')

        outputs = {}
        permuteVars = self._var_permute()

        for var in permuteVars:
            self.inputDict['var'] = var
            outputs[var] = ols_model(self.pre, self.post, self.inputDict)

        self._modelPool = outputs
        self._pool_stats()

        print('... run_all_linear() complete')

    def _pool_stats(self):
        """ Grabs specific stats for each model and stores them in a dataframe

        Returns
        -------
        string

        Raises
        ------
        AssertionError - Must run all models before diving into the pool
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

            for var, mod in self._modelPool.iteritems():
                modelNumber += 1

                newStatsRow = {'var': var,
                               'AIC': mod.Fit.aic,
                               'R2': mod.Fit.rsquared,
                               'AR2': mod.Fit.rsquared_adj,
                               'cvrmse': mod.Fit.cvrmse,
                               'postDiff': mod.postDiffSum}

                statsPool[modelNumber] = newStatsRow

        except AttributeError:  # py27 - dict.iteritems() py36 - dict.items()
            for var, mod in self._modelPool.items():
                modelNumber += 1

                newStatsRow = {'var': var,
                               'AIC': mod.Fit.aic,
                               'R2': mod.Fit.rsquared,
                               'AR2': mod.Fit.rsquared_adj,
                               'cvrmse': mod.Fit.cvrmse,
                               'postDiff': mod.postDiffSum}

                statsPool[modelNumber] = newStatsRow

        # Build DataFrame
        self.statsPool = pd.DataFrame(statsPool).T.sort_values('AIC')
        # Re-order columns
        self.statsPool = self.statsPool[['AIC', 'AR2', 'R2', 'cvrmse',
                                         'postDiff', 'var']]

    def plot_pool(self, number=5):
        """ Plots the top 5 models after run_all"""
        for i in range(number):
            modVars = self.statsPool['var'].iloc[i]
            tempMod = self._modelPool[modVars]

            tempMod.model_plot2()
            tempMod.stats_plot()
            plt.show()


class remodel():

    """This class opens a saved 'archived' model and recombines all of the
    input parameters to create the ols_model again. Once reborn, this class
    can also extend the data using PI to calculate more of the post period
    including, savings

    """

    @staticmethod
    def read_json_params():
        """ Read previous models parameters from the saved json file"""
        # read in Json params locally, split into two dicts
        # return two objects

        jsonName = 'all Params.json'
        with open(jsonName) as f:
            data = json.load(f)

        return data

    @staticmethod
    def read_raw_data():
        """ Read previous model's rawData file """
        # find raw data locally, load into pd.dataframe
        # return data

        fileName = 'raw data.xlsx'  # TODO: File name will be flexible (right?)
        df = pd.read_excel(fileName, index_col=0,
                           parse_dates=True, infer_datetime_format=True)

        return df

    @staticmethod
    def extend_data_with_pi(data, endDate='y'):
        """ Grabs more PI data for the commodity of interest up to the
        specified date

        """
        # Find last date in data, make that start date for datapull
        startDate = data.index[-1] + pd.Timedelta('1h')
        # Default end date to yesterday

        # Use column headings as tags
        tags = list(data.columns)

        # Pull data
        # Refactor a single pi_client into the data stuff, source? ?
        newData = pi.get_stream_by_point(tags, start=startDate,
                                         end=endDate, interval='1h')
        # TODO: add PI failure exception

        # concat and return
        combinedData = pd.concat([data, newData], axis=0)

        return combinedData

    @staticmethod
    def DK_II(data, dataParams):
        """
        My numpydoc description of a kind
        of very exhautive numpydoc format docstring.

        Parameters
        ----------
        # TODO: WRITE ME first : array_like
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
        # Change the self.postEnd variable to the most recent day in the data
        dataParams['dateRanges'][3] = data.index[-1].strftime('%Y-%m-%d')
#        dataParams['IQRmult'] = 6

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
        # TODO: WRITE ME first : array_like
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

        print(modelParams)

        mc = ols_model(pre, post, modelParams)

        mc.calculate_kfold()
        mc.calculate_vif()

        return mc


# =============================================================================
# --- Functions
# =============================================================================

def _create_new_save_directory(dirName=None, useDate=False, _index=None):
    """Create new directory where the name and date and be specified
    New directories are created recursively with a (#) appended to the name if
    there is a name match

    Parameters
    ----------
    dirName : str
        the name of the directory to make
    useDate : Bool (False)
        True - Adds the '%Y-%m-%d_%H%M%S' strftime to the end of the dirName
        False - only uses the dirName to make the new folder
    _index : int (None)
        Used to recusrively keep track of which folder number needs to be made
        if this folder name is taken

    Returns
    -------
    None - Creates a directory

    Raises
    ------

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

# TODO: Make inverval checker only run if needed, this will be a work around to
    # avoid month issues. Or find a way that this function can recognize months

def _interval_checker(df, unit='min', silence=True, symbolOutput=False):
    """
    Copied from mypy on 08/23/2018 to reduce imports

    Calculate the intervalsize of a time series dataset and returns
    the mode() of the index timedeltas in specified unit eg: 60.0

    #TODO: Docstring me
    """

    # Unit must be min for dictionary to work
    if symbolOutput:
        unit = 'min'

    if isinstance(df, pd.DataFrame):
        if isinstance(df.index, pd.core.indexes.datetimes.DatetimeIndex):
            pass
        else:
            raise TypeError("df must have datetimeIndex")
    else:
        raise TypeError("input must be of type: pandas DataFrame")

    idx = df.index
    deltaList = [idx[i] - idx[i - 1] for i in range(1, len(idx))]
    s = pd.Series(deltaList)
    valueCounts = s.value_counts()

    ratio = float(valueCounts[0]) / float(valueCounts.sum())
    if ratio < 0.985:
        print("Warning: >1.5% of intverals not in mode() of intervals")
        print("{0:.2f}% ratio\n".format(100 * (1 - ratio)))
        print(valueCounts)

    if unit.lower() in ['m', 'min', 'minute', 'minutes']:
        denominator = 60
    elif unit.lower() in ['h', 'hr', 'hour', 'hours']:
        denominator = 3600
    elif unit.lower() in ['d', 'dy', 'day', 'days']:
        denominator = 86400

    timeDelta = valueCounts.index[0]
    output = timeDelta.total_seconds() / denominator

    if silence:
        pass
    else:
        print('Timedelta: "{}" is {} {}'.format(timeDelta, output, unit))

    if symbolOutput:
        # HACK: This symbolic output only works if unit = 'min' so above
        # The function forces unit to min
        symbolDict = {1: '1Min',
                      15: '15Min',
                      60: 'H',
                      1440: 'D',
                      44640: 'M',
                      43200: 'M'
                      }

        return symbolDict[int(output)]

    return output


def _build_time_columns(dfIn, daytime=(8, 20)):
    """
    Function copied from mypy on 08/23/2018 to reduce imports

    This function will add many columns to the end of the dataframe which
    contains information about the rows dateTime but split into columns

    year, month, dayofmonth, hour, minute, dayofweek, weekday, daytime,
    saturday, sunday
    """
    df = dfIn.copy()

    df['year'] = df.index.year
    df['month'] = df.index.month
    # added for mnv tool functionality
    df['month'] = df['month'].astype('category')
    df['dayofmonth'] = df.index.day
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['dayofweek'] = df.index.dayofweek
    df['weekofyear'] = df.index.week

    mask = (df['hour'] >= daytime[0]) & (df['hour'] < daytime[1])
    df['daytime'] = 0
    df.loc[mask, 'daytime'] = 1

    mask = df['dayofweek'] < 5
    df['weekday'] = 0
    df.loc[mask, 'weekday'] = 1

    mask = df['dayofweek'] == 5
    df['saturday'] = 0
    df.loc[mask, 'saturday'] = 1

    mask = df['dayofweek'] == 6
    df['sunday'] = 0
    df.loc[mask, 'sunday'] = 1

    return df


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
        completed['kfold'] = False
    try:
        assert(isinstance(mc.savingsSummary, pd.DataFrame))
        completed['savings'] = True
    except AttributeError:
        print('create_archive warning: generate_savings_summary has not been run')
        completed['savings'] = False
    try:
        assert(isinstance(mc.F, float))
        completed['uncertainty'] = True
    except AttributeError:
        print('create_archive warning: uncertainty has not been run')
        completed['uncertainty'] = False

    return completed


def _calculate_degree_hours(oatData=None, by='day', cutoff=65):
    """ Copied this function from mypy on 08/23/2018 to reduce imports
        '''
        Calculate the HDH or CDH in a day or month as a standalone action
        Source of data is always OAT master file
        '''
    """

    if isinstance(oatData, pd.DataFrame) or isinstance(oatData, pd.Series):
        if isinstance(oatData, pd.Series):
            df = oatData.to_frame()
        else:
            df = oatData
        df.columns = ['OAT']
        print('OAT supplied with df')
    else:

        oatPath = path.join(pathPrefix, 'OATmaster.csv')
        print('OAT being loaded from master file {}'.format(oatPath))

        df = pd.read_csv(oatPath, index_col=0, parse_dates=True,
                         infer_datetime_format=True)

    # always resample data to hour for HDH calcs
    df = df.resample('1H').mean()
    # Build HDH/CDH columns
    hours = pd.DataFrame()
    hours['HDH'] = cutoff - df['OAT']
    hours['CDH'] = df['OAT'] - cutoff
    hours.index = df.index
    hours[hours < 0] = 0

    hours['CDH2'] = hours['CDH'] ** 2
    hours['HDH2'] = hours['HDH'] ** 2

    # Resample again with sum() if needed
    if by.lower() == 'day' or by.lower() == 'd':
        hours = hours.resample('1D').sum()
    elif by.lower() == 'month' or by.lower() == 'm':
        hours = hours.resample('1M').sum()
    elif by.lower() == 'hour' or by.lower() == 'h':
        pass
    else:
        print("WARNING: calculate_degree_hours didn't get by=H, D or M")
    return hours


def create_archive(dk, mc,
                   saveFigs=False,
                   copyRemodel=False,
                   customDirectoryName=None,
                   centralize=False):
    """
    Saves many files to archive a data/model run.

    These items are used for compiling the M&V report
    and re-running models using the __Model__ notebook

    Parameters
    ----------
    dk : instance of data_keeper
        needed to extract the params
    mc : instance of the ols_model class
        needed to extract the vif/kfold/plots and params
    saveFigs : Bool (False)
        True - output selected plots as .pngs
        False - no output plots
    copyRemodel : Bool (False)
        True - Create a copy of the reMnV.ipynb in the saveDirectory folder
            :: used when creating a new model archive, not needed if using the
            :: reMnV.ipynb
        False - Do not copy the notebook template to saveDirectory
    customDirectoryName : str (None)
        If string supplied saveDirectory will be named customDir + date
        if None - dk.com is used as folder name (dk.com)
    centralize : bool (False)
        Change to True when creating the end of fiscal year central results
        goals:
            1. save all savings data in a central spot?
            2. Make a pdf for each line item

            3. TODO: other?

    Returns
    -------
    None - However it may output a .png(s) .xlsx .json and .ipynb in a new dir

    Raises
    ------
    None
    """

    if centralize:
        centralDirectory = 'REPORT'
        dirName = '{}/{}/{}'.format('..', centralDirectory, dk.com)
    else:
        dirName = dk.com

    if customDirectoryName:
        saveDirectory = _create_new_save_directory(dirName=customDirectoryName,
                                                   useDate=True)
    else:
        saveDirectory = _create_new_save_directory(dirName=dirName,
                                                   useDate=True)

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
        mc.kfoldStats.to_excel(writer, sheet_name='Kfold')

    if completionDict['savings']:
        mc.savingsSummary.to_excel(writer, sheet_name='savings')

    if completionDict['uncertainty']:
        df = pd.DataFrame(index=['Fraction Savings', 'Uncertainty'],
                          data=[mc.F, mc.uncertainty])
        df.to_excel(writer, sheet_name='uncertainty')

    # write ols_model.Fit().summary() in sections
    # dfList is 3 seperate stats summaries, so we need to write them to the
    # same sheet iteratively and increment the startrow by the shape + 1
    dfList = pd.read_html(mc.Fit.summary().as_html())
    startRow = 0

    for df in dfList:
        df.to_excel(writer, sheet_name='Summary', startrow=startRow,
                    startcol=0, header=False, index=False)
        startRow += df.shape[0] + 1

    # Write all params into a json for reading .
    # Necessary to access __dict_ since it contains the default/hidden params
    allParams = {'dataParams': dk.params.__dict__,
                 'modelParams': mc.params.__dict__}

    with open('{}/all Params.json'.format(saveDirectory), 'w') as f:
        json.dump(allParams, f, indent=4)

    dk.rawData.to_excel('{}/raw data.xlsx'.format(saveDirectory))

    if copyRemodel:
        sourceName = 're-Model-MnV.ipynb'

        copyfile(path.join(pathPrefix, sourceName),
                 path.join(saveDirectory, sourceName))

    # Output images
    if saveFigs:
        mc.mod1Plot.savefig('{}/Model plot.png'.format(saveDirectory))
        mc.savingsPlot.savefig('{}/Savings plot.png'.format(saveDirectory))
        mc.tmyPlot.savefig('{}/TMY plot.png'.format(saveDirectory))
