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


Last update 6/12/18 - koshnick
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

from sklearn.model_selection import train_test_split, KFold
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
#from sklearn.metrics import mean_squared_error, r2_score

#path_prefix = path.dirname(path.abspath(__file__))

sys.path.append('../../mypy')
plt.rcParams.update({'figure.max_open_warning': 0})

figW = 18
figH = 6

version = 'Version 1.2'

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
    data : pd.DataFrame
        Timeseries data
    inputDict : dict
        **kwargs type dict to be passed into the DataParameters class
    
    Returns
    -------
    instance: dc
        notable attributes: dc.pre, dc.post
    
    Raises
    ------
    ## TODO: Write me
    KeyError
        when a key error
    OtherError
        when an other error
    """
    
    """
    input **kwargs

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
        ## TODO: rename self.com to something more meaning
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


    def interval_checker():
        """
        Calculate the intervalsize of a time series dataset and returns
        'hourly', '15 minute', 'daily', 'monthly'
        """
        ## TODO: PROGRAM ME
        pass

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



    ## DOC STRING END
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
        we're using for the M&V (usually daily) ##CHECK make sure this works
        if we want to make an hourly model

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
        self.restoreData = self.modifiedData.copy()
        ## Check redundant code - refactor


        # TODO: Should not use resample rate here, should use actual invterval
        # assert that it is H or D??
        if self.params.OATsource == 'file':
            hours = mypy.calculate_degree_days(oatData=None,
                                               by=self.params.resampleRate,
                                               cutoff = 65)
            
        elif self.params.OATsource == 'self':
            hours = mypy.calculate_degree_days(oatData=self.OAT,
                                               by=self.params.resampleRate,
                                               cutoff = 65)

        self.modifiedData['HDH'] = hours['HDH']
        self.modifiedData['CDH'] = hours['CDH']

        self.modifiedData['HDH2'] = self.modifiedData['HDH'] ** 2 ## Check. How do we sum HDH2 vs HDD2?
        self.modifiedData['CDH2'] = self.modifiedData['CDH'] ** 2

        return self

    def add_time_columns(self):
        self.modifiedData = mypy.build_time_columns(self.modifiedData)

    def add_dummy_variables(self):

        
        #TODO: Add code to take ~ C(month) params and split into 'month'
        # Only using dummies to make VIF work at this point
        dummyColumns = ['month','weekend','weekday','hour']
        dums = pd.get_dummies(self.modifiedData[dummyColumns],
                              columns=dummyColumns,
                              drop_first=True)
        
        
        self.modifiedData = pd.concat([self.modifiedData,dums],axis=1)
        pass


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
#        self.add_degree_hours()
        self._resample(aggFun = 'sum')
        self.add_degree_hours()
        self.add_time_columns()
#        self.add_dummy_variables()
        self.data_slice()


#==============================================================================
# PLOTTING
#==============================================================================

    def _outlier_plot(self, temp, yrange=None, title=None):

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
        
    def _pre_post_plot(self):
        fig = plt.figure(figsize=(figW, figH))
        ax = fig.add_subplot(111)
        ax.plot(self.pre[self.com], color='b')
        ax.plot(self.post[self.com], color='g')
        plt.legend(['pre','post'])
        plt.title('Pre and Post period')
        plt.ylabel(self.com)
        plt.show()
        

#==============================================================================
# MODELING
#==============================================================================

class ModelParameters():
    """
    My numpydoc description of a kind
    of very exhautive numpydoc format docstring.
    
    Parameters
    ----------
    first : array_like
        the 1st param name `first`

    inputDict = {'params' : 'C(month) + C(weekend)',
         'testTrainSplit' : 'random',
         'randomState': 4291990,
         'testSize': 0.2,
         'commodityRate' : 0.056,
         'paramList' : ['','C(month)','C(weekend)']}
    
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
    
    def __init__(self,
                 params = 'C(month) + C(weekend)',
                 testTrainSplit = 'random',
                 randomState=4291990,
                 testSize=0.2,
                 commodityRate=0.056,
                 paramList=['', 'C(month)', 'C(weekend)']):
    
        self.params = params
        self.testTrainSplit = testTrainSplit
        self.randomState=randomState
        self.testSize=testSize
        self.commodityRate=commodityRate
        self.paramList = paramList

    def show_params(self):
        """ Display the key, value pairs of the parmeters in this class"""
        for k,v in self.__dict__.iteritems():
            print(k,v)

class ols_model():
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
               
        self.params = ModelParameters(**inputDict)
        self.pre = pre
        self.post = post
        self.com = pre.columns[0]

        # TODO: Implement **kwargs input dict for models?
        
        self.split_test_train(how=self.params.testTrainSplit)
        self.Model = smf.ols(self.com + '~' + self.params.params, data=self.train)
        self.Fit = self.Model.fit()
        
        self.Fit.cvrmse = math.sqrt(self.Fit.mse_resid) / self.pre[self.com].mean()

        #Make predictions called "Calcs"
        self.trainCalc = self.Fit.predict(self.train)
        self.testCalc = self.Fit.predict(self.test)
        self.postCalc = self.Fit.predict(post)
        #            try:
#                self.inputDict['params'] = params
#                outputs[params] = ols_model(self.pre, self.post, self.inputDict)
        
        try:
            self.calculate_vif()
        except Exception as e:
            print('Could not calculate VIF for {}'.format(self.params.params))
            print('Exception caught: {}'.format(e))
            self.vif = None
            
        # XXX: does this work? what is fit.resid?
        self.postDiff = self.postCalc - self.post[self.com]
        self.postCumsum = self.postDiff.cumsum()[-1]
     
    @staticmethod
    def get_dummy_strings(params):
        dumList = ['year','month','day','weekday','weekend','hour','daytime']
        outList = []
        
        for dum in dumList:
            if params.find("("+dum+")") > 0:
                outList.append(dum)
        
        return outList

            
    @staticmethod
    def get_columns_from_params(params):
        
        splitStrings = params.split('+')
        
        for i, splitString in enumerate(splitStrings):
            splitString = splitString.replace("C(","")
            splitString = splitString.strip(' ()')
            splitStrings[i] = splitString
        
        return splitStrings
    
    @staticmethod
    def _degree_day_filter(params):
        
        dumList = ['CDH','HDH','CDH2','HDH2']
        splitStrings = params.split('+')
        
        for i, st in enumerate(splitStrings):
            splitStrings[i] = st.strip(' ')
            
#        print(params)
#        print(splitStrings)
        
        missList = []
        
        for dum in dumList:
            if dum in splitStrings:
                pass
            else:
                missList.append(dum)
                
        return missList
    
    
    def calculate_vif(self):
        
        #TODO: Add code to take ~ C(month) params and split into 'month'
        # Only using dummies to make VIF work at this point
        dummyColumns = ols_model.get_dummy_strings(self.params.params)
        
#        print(dummyColumns)
        
        if dummyColumns == []:
            dums = None
        else:
            dums = pd.get_dummies(self.train[dummyColumns],
                      columns=dummyColumns,
                      drop_first=True)
        
        if isinstance(dums, pd.DataFrame):
            testdf = pd.concat([self.train,dums],axis=1)
        else:
            testdf = self.train
        
        testdf = mypy.remove_time_cols(testdf)
        missList = ols_model._degree_day_filter(self.params.params)
        
#        print(missList)
        
        testdf = testdf.drop([self.com] + missList, axis=1)
        testdf = add_constant(testdf)
        
#        print(testdf.head())
        
        vif_values = [variance_inflation_factor(testdf.values, i) for i in range(testdf.shape[1])]
        
        vif = pd.DataFrame(index=testdf.columns,data=vif_values,columns=['VIF'])
    
        self.vif = vif


    def split_test_train(self, how='random', testSize=None, randomState=None):
        """
        Take the pre data set and create two variables "test" and "train" to
        feed into the model

        Split the data in the following ways:
            -1/3 : 2/3 simple split
            -a random collection of 1/3 : 2/3
            -as a Kfold with or without shuffling
        """
        
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

            ## XXX: COMMENT THIS
            splits = train_test_split(self.pre,
                                      test_size=testSize,
                                      random_state=randomState)
            
            self.train = splits[0]
            self.test = splits[1]
            pass


    def model_plot(self):

        fig = plt.figure(figsize=(figW, figH*1.5))
        ax0 = plt.subplot2grid((2,1), (0,0))

        ax1 = plt.subplot2grid((2, 1), (0, 0), colspan=1)
        self.test[self.com].plot(label='actual')
        self.testCalc.plot(label='model')

        plt.title('Test data ' + self.params.params)
        plt.ylabel(self.com)
        plt.legend()

        ax2 = plt.subplot2grid((2, 1), (1, 0), colspan=1)

        self.post[self.com].plot(label='actual')
        self.postCalc.plot(label='model')
        plt.ylabel(self.com)
        plt.title('Post data ' + self.params.params)
        plt.legend()
        plt.tight_layout()
        
    def model_plot2(self):

        fig = plt.figure(figsize=(figW, figH))
        ax0 = plt.subplot2grid((1, 1), (0, 0))

        ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=1)
        self.test[self.com].plot(label='actual')
        self.testCalc.plot(label='model')

        plt.title('Test data ' + self.params.params)
        plt.ylabel(self.com)
        plt.legend()


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

        #figure
        fig = plt.figure(figsize=(figW*1.2, figH*1.5))
        ax0 = plt.subplot2grid((3,1), (0,0))
        
        #plot 0
        ax1 = plt.subplot2grid((3, 1), (0, 0), colspan=1)

        self.post[self.com].plot(label='actual')
        self.postCalc.plot(label='model')
        plt.ylabel(self.com)
        plt.title('Post data ' + self.params.params)
        plt.legend()
        
        #plot 1
        ax1 = plt.subplot2grid((3, 1), (1, 0), colspan=1)

        savingsPos = ydata[ydata >= 0]
        savingsNeg = ydata[ydata < 0]

        plt.plot(savingsPos, color = 'k', linestyle='',marker='.', markersize=pointSize)
        plt.plot(savingsNeg, color = 'r',linestyle='',marker='.', markersize=pointSize)
#        self.postTest.plot(label='model')
        plt.title('Savings predicted by ' + self.params.params)
        plt.ylabel('Savings {}'.format(ylab))
        plt.legend()

        #plot 2
        ax2 = plt.subplot2grid((3, 1), (2, 0), colspan=1)

        cumulative = ydata.cumsum()

        cumPos = cumulative[cumulative >= 0]
        cumNeg = cumulative[cumulative < 0]

        plt.plot(cumPos, color = 'k', linestyle='',marker='.', markersize=pointSize)
        plt.plot(cumNeg, color = 'r', linestyle='',marker='.', markersize=pointSize)        
        plt.ylabel('Cumulative Savings')
        plt.legend()
        
        plt.tight_layout()
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

    @staticmethod
    def make_param_string(columnCategories):
        params = ''

        if 'month' in columnCategories:
            params += 'month_1+month_2+month_3+month_4+month_5+month_6+month_7+month_8+month_9+month_10+month_11+month_12+'

        if 'weekday' in columnCategories:
            params += 'weekday_0+weekday_1+weekday_2+weekday_3+weekday_4+weekday_5+weekday_6+'

        if 'weekend' in columnCategories:
            params += 'weekend_0+weekend_1'

        if 'hour' in columnCategories:
            params += 'hour_0+hour_1+hour_2+hour_3+hour_4+hour_5+hour_6+hour_7+hour_8+hour_9+hour_10+hour_11+hour_12+hour_13+hour_14+hour_15+hour_16+hour_17+hour_18+hour_19+hour_20+hour_21+hour_22+hour_23+'

        return params.rstrip('+')




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

        if self.inputDict['paramList'] == None:
            inputs = ['', 'C(month)','C(weekend)']
        else:
            inputs = self.inputDict['paramList']

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