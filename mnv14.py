# -*- coding: utf-8 -*-
"""
Created on Wed Apr 04 10:07:41 2018

@author: koshnick

Last update 5/14/18 - koshnick
Added Params subclass to data container. Refactored code to fit new convention
Next time - need to add internal functions to _convention.

v1.2 6/12/2017

- remove postCalc graph from model_plot() since this plot is not needed to eval
    uate the quality of the model. postCalc plot now shown in savings plot.
- added cvrmse stat as mod.Fit.cvrmse
- removed MSE from stats_pool (in favor of cvrmse)
- added calculate_vif function and stored result under mod.vif
- Fixed the ability to use OAT within the same dataset, however it is clunky
- Added custom markersize variable for savings plot
- Added ModelParameters class to allow for a modelDict input for the model class

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
- Fixed issue where calculate_vif() would throw error MissingDataError: exog contains inf or nans
-- The OAT data was creating HDH/CDH values which were nan and causing this error

v1.4 7/6/2018
- Updated Kfold and split_test_train so now the train and test for the main model are identical
    to fold 0 in the Kfold analysis. Added variable self._folds that is a collection of indicies
    generated from sklearn KFold(). This is used to now generate test_train, and reused to calc
    the Kfold stats
- Fixed conversion of paramList to paramString in ols_model's
     __init__. It was formerly dropping HDH CDH HDH2 and CDH2

known issues:


Last update 6/25/18 - koshnick
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

#from os import path
from datetime import datetime
#from sklearn import linear_model
#from matplotlib.gridspec import GridSpec

from sklearn.model_selection import KFold
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
#from sklearn.metrics import mean_squared_error, r2_score

#path_prefix = path.dirname(path.abspath(__file__))

sys.path.append('../../mypy')
plt.rcParams.update({'figure.max_open_warning': 0})

figW = 18
figH = 6

version = 'Version 1.4'

class DataParameters():
    """
    A simple class to store attributes for the data_handler class. The __init__
    function is specially formated for the data_handler class and takes in a
    **kwargs dictionary to populate the parameters. All parameters have default
    values, but can be modified in **kwargs.

    Parameters
    ----------
    **inputDict : dict
        each entry in the dictionary should map to a variable in the __init__

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

    IQRmult : float
        Number that the IQR will be bultiplied by to adjust floor/ceiling
    IQR : str [either 'y' or None]
        if IQR == 'y' then the outlier detection will use IQR method
    floor : int
        Lowest value any data point should take on

    Returns
    -------
    class instance
        instance.attibutes

    Raises
    ------
    None
    """

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

    def show_params(self):
        """ Display the key, value pairs of the parmeters in this class"""
        for k,v in self.__dict__.iteritems():
            print(k,v)

class data_handler():
    """
    Multipurpose data cleaning instance. Designed to take in a pd.DataFrame,
    modify it in several ways, cut the data, and pass it off for M&V modeling.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Timeseries data
    inputDict : dict
        **kwargs type dict to be passed into the DataParameters class

    Returns
    -------
    instance: dc
        notable attributes
        ------------------
        dc.modifiedData - working data object
        dc.restoreData - a copy made before modifiedData is changed
        ## TODO: Should I make an option to shut this off to increase speed
        dc.pre - pre period data created by data_slice()
        dc.post - post period data created by  data_slice()


    Raises
    ------
    ## TODO: Write me
    KeyError
        when a key error
    OtherError
        when an other error
    """

    # Data Class init
    def __init__(self, data, inputDict=None):
        # instantiate params
        self.params = DataParameters(**inputDict)
        self.params.modifiedDataInterval = 'raw'

        # Allow for pd.series or pd.dataframe to be loaded. In the case of a
        # dataframe, use the first column as the data of interest, drop others
        if isinstance(data,pd.DataFrame):
            self.rawData = data[data.columns[0]].to_frame()
            print('Loaded pd.DataFrame - using column "{}"'.format(data.columns[0]))
        elif isinstance(data,pd.Series):
            self.rawData = data.to_frame()

        # Use the OAT loaded into instance if present, otherwise OAT data will
        # be loaded from OAT master
        if self.params.OATsource == 'self':
            self.OAT = data[self.params.OATname]

        # modifiedData is the working variable for all of the data modifications
        self.modifiedData = self.rawData.copy()

        ## TODO: rename self.com to something more meaningful
        self.com = self.modifiedData.columns[0]

        ## TODO: implement archiving since that is what date and name are for
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
        ## TODO: Should this function plot? on verbosity?
        '''
        self.modifiedData = self.restoreData.copy()


#==============================================================================
# CLEANING
#==============================================================================

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
        self.restoreData = self.modifiedData.copy()

        #temp needed for being temporary modified data and _outlier_plot
        temp = self.modifiedData.copy()

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

        # Select data where floor < data < ceiling
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
#        return self ## CHECK do i need this here?


    def add_oat_column(self):
        """
        Call mypy to add OAT data to the commodity data. Only need to call this
        if the input data was not supplied with assocaited OAT data.
        """
        # Store copy incase self.undo() invoked
        self.restoreData = self.modifiedData.copy()

        if self.OATsource == 'file':
            ##CHECK rework MYPY and this function
            self.modifiedData = mypy.merge_oat(self.modifiedData,
                                               choose='y')

        elif self.OATsource == 'self':
            self.modifiedData['OAT'] = self.OAT

        print('Created OAT from {}'.format(self.OATsource))
        print('')

#        return self ##CHECK Do i even need this?


    def _resample(self, resampleRate=None, aggFun = 'mean'):

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

        self.restoreData = self.modifiedData.copy()

        if resampleRate == None:
            resampleRate = self.params.resampleRate

        self.modifiedData = self.modifiedData.resample(resampleRate).agg(aggFun)
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
        self.restoreData = self.modifiedData.copy()


        # TODO: Should not use resample rate here, should use actual invterval
        # assert that it is H or D??
        if self.params.OATsource == 'file':
            hours = mypy.calculate_degree_hours(oatData=None,
                                               by=self.params.resampleRate,
                                               cutoff = cutoff)

        elif self.params.OATsource == 'self':
            hours = mypy.calculate_degree_hours(oatData=self.OAT,
                                               by=self.params.resampleRate,
                                               cutoff = cutoff)

        self.modifiedData['HDH'] = hours['HDH']
        self.modifiedData['CDH'] = hours['CDH']

        self.modifiedData['HDH2'] = self.modifiedData['HDH'] ** 2 ## Check. How do we sum HDH2 vs HDD2?
        self.modifiedData['CDH2'] = self.modifiedData['CDH'] ** 2

        # Drop any row where null values appear
        self.modifiedData = self.modifiedData.dropna(axis=0, how='any')

        return self

    def add_time_columns(self):
        self.modifiedData = mypy.build_time_columns(self.modifiedData)

    def add_dummy_variables(self):


        #TODO: Add code to take ~ C(month) params and split into 'month'
        # Only using dummies to make VIF work at this point
        dummyColumns = ['month','weekday','dayofweek','hour']
        dums = pd.get_dummies(self.modifiedData[dummyColumns],
                              columns=dummyColumns,
                              drop_first=True)


        self.modifiedData = pd.concat([self.modifiedData,dums],axis=1)
        pass


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
        ## TODO: Check if the OAT resample time mean and sum all work together correctly

        self.remove_outliers()
        self._resample(resampleRate = 'H', aggFun = 'mean')
        self._resample(aggFun = 'sum')

        self.add_degree_hours()
        self.add_time_columns()
        self.data_slice()


#==============================================================================
# PLOTTING
#==============================================================================

    def _outlier_plot(self, temp, yrange=None, title=None):
        """
        This plotting method allows the user to see which outliers are being
        removed from the dataset, and a statistics comparison in the form of a
        boxplot. Red datapoints indicated points to be removed.

        Parameters
        ----------
        temp: pd.DataFrame
            This is the data that will be removed from self.modifiedData
        yrange : tuple
            if wanting to constrain the yrange to (ymin,ymax)
        title : str
            custom title string -- probably useless

        Returns
        -------
        None - only plots

        Raises
        ------
        None

        """

        fig = plt.figure(figsize=(figW, figH))
        ax0 = plt.subplot2grid((1,5), (0,0))

        ## Box plot
        ax1 = plt.subplot2grid((1, 5), (0, 0), colspan=1)

        _comb = pd.concat([self.modifiedData[self.com],temp], axis=1)
        _comb.columns = ['before','after']
        sns.boxplot(data=_comb, ax=ax1)
        plt.ylabel(self.com)
        plt.title('Before and After')

        ## Scatter plot
        ax2 = plt.subplot2grid((1, 5), (0, 1), colspan=4)

        indexDifference = self.modifiedData.index.difference(temp.index)

        plt.plot(temp, color= 'k', linestyle='', marker='.')
        plt.plot(self.modifiedData[self.com][indexDifference], color = 'r',
                 linestyle='', marker='.')

        plt.title('Outlier removal result. interval = {}'.format(
                str(self.params.modifiedDataInterval)))

#        self.params.modifiedDataInterval = resampleRate
        if yrange:
            plt.ylim(yrange)
        plt.show()


    def _pre_post_plot(self):
        fig = plt.figure(figsize=(figW, figH))
        ax = fig.add_subplot(111)
        ax.plot(self.pre[self.com], color='royalblue')
        ax.plot(self.post[self.com], color='forestgreen')
        plt.legend(['pre','post'])
        plt.title('Pre and Post period')
        plt.ylabel(self.com)
        plt.show()


#==============================================================================
# MODELING
#==============================================================================

class ModelParameters():

    """
    A simple class to store attributes for the ols_model class. The __init__
    function is specially formated for the data_handler class and takes in a
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
        '## TODO' - Are there other methods to do this?
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
                 params = ['month', 'weekday'],
                 testTrainSplit = 'random',
                 randomState=4291990,
                 testSize=0.2,
                 commodityRate=0.056,
                 paramPermuteList=['', 'C(month)', 'C(weekday)']):


        #Will be changed to not None with _convert_params
        self.paramList, self.paramString = None, None
        self._convert_params(params)

#        self.params = params #Goodbye params
        self.testTrainSplit = testTrainSplit
        self.randomState=randomState
        self.testSize=testSize
        self.commodityRate=commodityRate
        self.paramPermuteList = paramPermuteList

    def _convert_params(self, params):


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
            #conver to string, paramList

        elif isinstance(params,str):
            paramList = params.split('+')
            paramList = [x.replace("C(","").replace(")","").strip(' ') for x in paramList]
            self.paramList = paramList
            self.paramString = params

            #convert to list, paramString
        else:
            raise ValueError('Params input must be either string or list of the form ["month","hour"] or "C(month) + C(hour)"')


    def show_params(self):
        """ Display the key, value pairs of the parmeters in this class"""
        for k,v in self.__dict__.iteritems():
            print(k,v)

class ols_model():
    """
    Designed to take inputs from data_handler and a **kwargs inputDict
    Performs a single linear regression based on input params, calculates
    predicted values using the self.post data set, and reports statistics
    information about the model

    Parameters
    ----------
    pre : pd.DataFrame
        Timeseries data passed from data_handler
    post : pd.DataFrame
        Timeseries data passed from data_handler
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
        self.com = pre.columns[0]

        self.split_test_train(how=self.params.testTrainSplit)


        self.Model = smf.ols(self.com + '~' + self.params.paramString, data=self.train)
        self.Fit = self.Model.fit()

        self.Fit.cvrmse = math.sqrt(self.Fit.mse_resid) / self.pre[self.com].mean()

        #Make predictions called "Calcs"
        self.trainCalc = self.Fit.predict(self.train)
        self.testCalc = self.Fit.predict(self.test)
        self.postCalc = self.Fit.predict(post)




#                    try:
#                self.inputDict['params'] = params
#                outputs[params] = ols_model(self.pre, self.post, self.inputDict)

        try:
            self.calculate_vif()
        except Exception as e:
            print('Could not calculate VIF for {}'.format(self.params.paramString))
            print('Exception caught: {}'.format(e))
            self.vif = None

        # XXX: does this work? what is fit.resid?
        self.postDiff = self.postCalc - self.post[self.com]
        self.postCumsum = self.postDiff.cumsum()[-1]


    def _remove_degree_days(self, params):

        params = list(params) #Ghetto copy
        dumList = ['CDH','HDH','CDH2','HDH2']
        removed = []

        for dum in dumList:
            if dum in params:
                params.remove(dum)
                removed.append(dum)
            else:
                pass


        return params, removed


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
        #Get dummies
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

        #join dummies to commodity and degree days
        testdf = pd.concat([self.pre[removed], dums],axis=1)
        # Add constant column if needed (essentially this is the intercept)

#        print(testdf.columns)
#        for dum in dums:
#            if len(testdf[dum].unique()) == 1:
#                del testdf[dum]
#        print(testdf.head())

        testdf = add_constant(testdf) #VIF Expects a constant column

        #Get VIF values
        vif_values = [variance_inflation_factor(testdf.values, i) for i in range(testdf.shape[1])]
        # Format into nice pd.DataFrame
        vif = pd.DataFrame(index=testdf.columns,data=vif_values,columns=['VIF'])

        self.vif = vif

        return vif


    def kfold(self):
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

        # model all folds
        # record MSE
        # find average MSE
        # find dataset deviation from MSE
        # find percent of MSE deviation
        # add to stats item?

#        # Calculate # of folds from 1/testSize
#        folds = int(round(1/self.params.testSize))
#        print(self.params.testSize, 'nfolds', folds)
#
#        #Use the scikitlearn kfolds function to randomize generate folds
#        kf = KFold(n_splits = folds, shuffle=True, random_state=self.params.randomState)

        #dict to hold final vales and transform to df
        statsPool = {}
        foldNumber = -1

        for train_index, test_index in self._folds:

            foldNumber += 1

            #split into test/train based on kfold indicies
            train = self.pre.iloc[train_index, :]
            test = self.pre.iloc[test_index, :]

            #TODO make sur ethat the randomstate in main params gives a fold
            # in this function to maintaine consitency
            Fit = smf.ols(self.com + ' ~ ' + self.params.paramString, train).fit()
            ## TODO: DO we need to be using predicted or Fit? MATH IS HARD
            predicted = Fit.predict(test)

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
        self.kfoldStats = kfoldStats[['R2','AR2','mse']]

        # Building relative MSE values
        mseMean = round(kfoldStats['mse'].mean())
        kfoldRelative = kfoldStats['mse']/mseMean*100
        kfoldRelative = kfoldRelative.append(pd.Series({'mse':mseMean}))
        self.kfoldRelative = kfoldRelative


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
        pd.DataFrame
            results of splitting generate two dataframes and are stored in
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
#
#
            self.train = self.pre.iloc[self._folds[0][0]]
            self.test = self.pre.iloc[self._folds[0][1]]

            pass

    def _get_folds(self,randomState):

        # find testsize split
        # make folds all randomnesses

        # Calculate # of folds from 1/testSize
        folds = int(round(1/self.params.testSize))
#        print(self.params.testSize, 'nfolds', folds)

        #Use the scikitlearn kfolds function to randomize generate folds
        kf = KFold(n_splits = folds,
                   shuffle=True,
                   random_state=randomState)

        self._folds = [x for x in kf.split(self.pre)]



    def model_plot(self):

        _color = 'mediumorchid'
        _style = '--'

        fig = plt.figure(figsize=(figW, figH*1.5))
        ax0 = plt.subplot2grid((2,1), (0,0))

        ax1 = plt.subplot2grid((2, 1), (0, 0), colspan=1)
        self.test[self.com].plot(label='actual',color='k')
        self.testCalc.plot(label='model',color=_color, linestyle=_style)

        plt.title('Test data ' + self.params.paramString)
        plt.ylabel(self.com)
        plt.legend()

        ax2 = plt.subplot2grid((2, 1), (1, 0), colspan=1)

        self.post[self.com].plot(label='actual', color='k')
        self.postCalc.plot(label='model',color=_color, linestyle=_style)
        plt.ylabel(self.com)
        plt.title('Post data ' + self.params.paramString)
        plt.legend()
        plt.tight_layout()

    def model_plot2(self):

        _color = 'mediumorchid'
        _style = '--'

        fig = plt.figure(figsize=(figW, figH))
        ax0 = plt.subplot2grid((1, 1), (0, 0))

        ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=1)
        self.test[self.com].plot(label='actual',color='k')
        self.testCalc.plot(label='model',color=_color, linestyle=_style)

        plt.title('Test data ' + self.params.paramString)
        plt.ylabel(self.com)
        plt.legend()


    # TODO: finish coding this function
    def stats_plot(self):

        fig = plt.figure(figsize=(figW*.8, figH * 0.75))
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


    def savings_plot(self, yaxis='raw', pointSize=4):

        if yaxis == 'raw':
            ydata = self.postDiff
            ylab = '[' + self.com + ']'

        elif yaxis == 'dollars':
            ydata = self.postDiff * self.params.commodityRate
            ylab = '[$]'

        else:
            raise ValueError('savings_plot requires y-axis to be == raw or dollars')
            return

        _color = 'mediumorchid'
        _style = '--'

        #figure
        fig = plt.figure(figsize=(figW*1.2, figH*1.5))
        ax0 = plt.subplot2grid((3,1), (0,0))

        #plot 0 - Model post
        ax1 = plt.subplot2grid((3, 1), (0, 0), colspan=1)

        self.post[self.com].plot(label='actual', color='k')
        self.postCalc.plot(label='model', color=_color, linestyle=_style)
        plt.ylabel(self.com)
        plt.title('Post data ' + self.params.paramString)
        plt.legend()

        #plot 1 - savings instant
        ax2 = plt.subplot2grid((3, 1), (1, 0), colspan=1)

        savingsPos = ydata[ydata >= 0]
        savingsNeg = ydata[ydata < 0]

        plt.plot(savingsPos, color = 'k', linestyle='',marker='.', markersize=pointSize)
        plt.plot(savingsNeg, color = 'r',linestyle='',marker='.', markersize=pointSize)
#        self.postTest.plot(label='model')
        plt.title('Savings predicted by ' + self.params.paramString)
        plt.ylabel('Savings {}'.format(ylab))
        plt.legend()

        #plot 2 - savings cumulative
        ax3 = plt.subplot2grid((3, 1), (2, 0), colspan=1, sharex=ax2)

        cumulative = ydata.cumsum()

        cumPos = cumulative[cumulative >= 0]
        cumNeg = cumulative[cumulative < 0]

        plt.plot(cumPos, color = 'k', linestyle='',marker='.', markersize=pointSize)
        plt.plot(cumNeg, color = 'r', linestyle='',marker='.', markersize=pointSize)
        plt.ylabel('Cumulative Savings')
        plt.legend()

        plt.tight_layout()
        fig.subplots_adjust(hspace=0)
        plt.show()


class many_ols():
    """
    Similar to the data holding class, will allow the user to put their data into
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
        self.inventory = None ## TODO: What is the variable?
        self.com = self.pre.columns[0]
        self.inputDict = inputDict


    def _param_permute(self):

        """
        Take in params and return a list of all combos of params without making
        stupid combos
        """

        a1 = ['CDH', 'CDH2', '']
        b1 = ['HDH', 'HDH2', '']

        if self.inputDict['paramPermuteList'] == None:
            inputs = ['', 'C(month)','C(weekday)']
        else:
            inputs = self.inputDict['paramPermuteList']

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

    def cross_valid_testing(self):
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

        """
        Make a function that tests if the model cross validates well using
        the k fold split of random data.

        Maybe this is the wrong class for this function?

        """

        pass


    def _pool_stats(self):

        """
        Take all of the models in modelPool and make sense of them so they
        can be ranked, plotted, etc..

        """

        try:
            assert(self._modelPool)
        except AssertionError:
            print('You must run "run_all_linear" to generate modelPool before'\
                  ' running pool_stats')

        statsPool = {}

        modelNumber = -1

        # Collect stats into dict of dicts
        for params, mod in self._modelPool.iteritems():
            modelNumber += 1

            newStatsRow = {'params' : params,
                           'AIC' : mod.Fit.aic,
                           'R2' : mod.Fit.rsquared,
                           'AR2' : mod.Fit.rsquared_adj,
                           'cvrmse': mod.Fit.cvrmse,
                           'postDiff' : mod.postCumsum,
                           'summary' : mod.Fit.summary()}

            statsPool[modelNumber] = newStatsRow

        # Build DataFrame
        self.statsPool = pd.DataFrame(statsPool).T.sort_values('AIC')
        # Re-order columns
        self.statsPool = self.statsPool[['AIC','AR2','R2','cvrmse','postDiff',
                                         'params','summary']]

    def plot_pool(self, number=5):

        for i in range(number):
            modParams = self.statsPool['params'].iloc[i]
            tempMod = self._modelPool[modParams]

            tempMod.model_plot2()
            tempMod.stats_plot()
            plt.show()



#==============================================================================
# TESTS
#==============================================================================

if __name__ == "__main__":
    pass