# -*- coding: utf-8 -*-
"""
Add a real doc string

@author: koshnick




update 6/18/2017
-Added interval checker
-moved functions to retired functions section:
    average_by_day
    group_slice
    plot_group_slice
    slice_and_plot
    pivot_and_plot_WITHOUT...
    standardize_columns
-Added improved functionality to pivot_and_plot:
    - color options (must be a Cmap)
    - ylim options
    - figsize option

Update 6/23/2018
- Changed names of time columns in build_time_columns
-- weekday -> dayofweek
-- weekend -> weekday
-- day -> dayofmonth
- Renamed build_degree_days to build_degree_hours since it truly calcs HDH and CDH
-- added deprecation warning to build_degree_days

known issues:

Last update 6/18/18 - koshnick
"""

import os
import string
import datetime
import itertools
# import matplotlib
import collections

import numpy as np
import pandas as pd


from os import path
from textwrap import wrap
import matplotlib.pyplot as plt

#from PI_client2 import pi_client
from PI_Client.v2 import *

path_prefix = path.dirname(path.abspath(__file__))

#buildingPath = path.join(path_prefix, 'data/Buildings.xlsx')
buildingPath = 'Buildings.xlsx'
buildings = pd.read_excel(buildingPath, index_col=0, header=0)


# This lets my exceptions play nice on py2.7 and py3.6
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

# =============================================================================
# --- Constants
# =============================================================================


class convert():
    """
    Convert from base units to kbtu and back again
    """
    # Electricity
    kwh_kbtu = 3.142
    kbtu_kwh = 1 / kwh_kbtu
    # Chilled Water
    tonh_kbtu = 12008
    kbtu_tonh = 1 / tonh_kbtu
    # Steam
    kbtu_klb = 1041
    klb_kbtu = 1 / kbtu_klb


class rate():
    """
    Utility rates last updated May 2018
    Convert energy to dollars in base units and kbtu (ACE RATES)
    """
    elec_kbtu = 0.0157   # USD / kbtu
    elec_kwh = 0.056     # USD / kWh

    chw_kbtu = 0.0030    # USD / kbtu
    chw_tonh = 0.036     # USD / tonh

    steam_kbtu = 0.0056  # USD / kbtu
    steam_klb = 5.667    # USD / klb


class colors():
    """
    CEED color codes in hex
    """
    chw = "#15A9C8"
    ele = "#98BF47"
    steam = "#F69321"
    gas = "#8168AE"
    solar = "#F8D81C"


class sample_data():
    """
    prescribed datasets to help troubleshoot functions
    """

    def time_series():
        """ Load simple time series ~ 4 months of 6 sensors hourly """
        return pd.read_excel(path.join(path_prefix, 'data/timeseries.xlsx'),
                             index_col=0, parse_dates=True)

    def rand(size=(30, 10)):
        """  Generate a dataframe of random values """
        letters = "".join(string.ascii_lowercase[0:size[1]])
        df = pd.DataFrame(np.random.randint(0, 50, size=size),
                          columns=list(letters))
        return df

    def ints():
        """ Make a small dataframe with unique integers in it """
        df = pd.DataFrame([[1, 2, 10],
                           [3, 4, 11],
                           [5, 6, 12],
                           [7, 8, 13]],
                          columns=["A", "B", "C"])
        return df

    def multiindex():
        """ Make a multi index with mixed type data """
        df = pd.DataFrame({'col1': {0: 'a', 1: 'b', 2: 'c'},
                           'col2': {0: 1, 1: 3, 2: 5},
                           'col3': {0: 2, 1: 4, 2: 6},
                           'col4': {0: 3, 1: 6, 2: 2},
                           'col5': {0: 7, 1: 2, 2: 3},
                           'col6': {0: 2, 1: 9, 2: 5},
                           })
        df.columns = [list('AAAAAA'), list('BBCCDD'), list('EFGHIJ')]
        return df


# =============================================================================
# --- Repickler
# =============================================================================


class repickler():

    """
    This class serves as a generalized handeler of archived PI data that is
    routinely updated and resaved. The top use case for this is to run monthly
    or weekly analysis on a static set of pi tags. The repickler can open the
    old data, update it with new data up until lastnight at midnight, then
    output the data as a pickle and/or .csv. It can also pass the data directly
    into an analysis script (not advised) using self.newData.

    inputs:
        picklePath - str:
            The FULLPATH of the pickle file that is going to be opened + saved
        piTags - None/list - default = None
            None if loading an old pickle
            list of pi tag strings if starting fresh
        csvOut = bool - default = False
            If True, save data as CSV
        interval - str - default = '15m':
            interval data for PI pull.
        newStart - str - default = '2018-01-01':
            if starting a new file, supply the startdate with newStart.
            the end date is defaulted to Yesterday
        newEnd - str - default = 'y':
            if starting a new pickle this will be the endDate of the first pull
    """
    # Start a pi client instance becuase we will be loading new data

    def __init__(self, picklePath,
                 tags=None,
                 interval='15m',
                 newStart='2018-01-01',
                 newEnd='y'):

        # Simple Vars
        self.updated = False
        self.new = False
        self.path = picklePath
        self.interval = interval
        self.newStart = newStart
        self.newEnd = newEnd

        # Calculate vars
        self.tags = tags  # CHECK make sure this handles old and new pickles
        self._loader(picklePath)
        self.csvName = str(picklePath).split('\\')[-1].rstrip('.pk1') + '.csv'

    '''
    def _tags(self, piTags):
        if isinstance(piTags, list):
            self.tags = piTags
        elif piTags == 'columns':
            self.tags = list(self.data.columns)
    '''

    def _loader(self, picklePath):
        """
        """

        try:
            self.data = pd.read_pickle(picklePath)
            if not self.tags:
                self.tags = list(self.data.columns)
        except FileNotFoundError:
            print('no such file as {}'.format(picklePath))
            print('starting new file at {}'.format(picklePath))
            print('will load data for pi tags starting at'
                  '{}'.format(self.newStart))

            self.new = True

    def update(self):
        """
        This function will load the historical archive of data from the pickle
        it will then look for the last date that was entered in the archive and
        pull new data (up until the start of today) using the tags in 'tags'
        Append the new data to the old data
        store new data  in self.updated
        """
        pi = pi_client()

        if self.new:
            startDate = self.newStart
            endDate = self.newEnd
        else:
            # Dont duplicate last entry, so move time to 1hour forward
            startDate = str((self.data.index[-1] +
                             pd.Timedelta(self.interval)).strftime('%Y-%m-%d'))
            # Make end date the begining of today
            endDate = str(datetime.datetime.now().strftime('%Y-%m-%d'))

        print('Beginning data pull from {} to {}'.format(startDate, endDate))

        self.newData = pi.get_stream_by_point(self.tags,
                                              start=startDate,
                                              end=endDate,
                                              interval=self.interval)

        if self.new:
            self.joinedData = self.newData
            print('shape of joinedData {}'.format(self.newData.shape))

        else:
            self.joinedData = pd.concat([self.data, self.newData],
                                        axis=0, join='outer')

            print('shape of olddata {}'.format(self.data.shape))
            print('shape of joinedData {}'.format(self.joinedData.shape))

        self.updated = True

    def save(self, csvOut=False):
        '''
        output local files if needed
        '''
        if not self.updated:
            print('pickle has not been updated, exiting save()')
        else:
            self.joinedData.to_pickle(self.path)
            if csvOut:
                self.joinedData.to_csv(self.csvName)

# =============================================================================
# --- PI extensions Functions
# =============================================================================


def search_tags(inputTags):
    """
    This function aims to make the pi.search_by_point() function more flexible.
    This function can take a string with * wildcards, or a list of strings with
    * wild cards. If a list is passed in, the function will concatenate all of
    the pi.search_by_point results into a single list while removing duplicates

    Inputs:
        inputTags (list or list of strings) - The search key for the PI tags of
        interest. Wildcard * are accepted.
    """

    # TODO: Add this function to the pi_client class instead of initializing it
    # just to run this function
    # Initalize PI web API client
    pi = pi_client()

    # Collector object for mulitple searches
    outputTags = []

    # Iterate through all tags in list of tags. Concat results in 'collector'
    if isinstance(inputTags, list):
        for tag in inputTags:
            searchedTags = pi.search_by_point(tag)[0]  # 0th element is tagname
            outputTags += searchedTags  # joining 2 lists

    # If input not a list, just search for that one tag
    elif isinstance(inputTags, str):
        outputTags = pi.search_by_point(inputTags)[0]  # 0th element is tagname

    # Remove duplicates and sort A->Z
    outputTags = sorted(list(set(outputTags)))

    return outputTags


def group_tags(allTags, parentLevel, sensorGroup):
    """
    This function will filter a list of tags to make sure a complete set of
    points are available. This helps when pulling data from PI to create an
    analysis of multiple points. If points A,B,C are being compared but point B
    does not exist, it will elimate that system from the group so time is not
    wasted pulling data for points A and C.

    The function operates by ensuring that a particular level in the name
    hierarchy (the parentLevel) has associated with it the points listed in
    sensorGroup.

    Consider the following example:

    If we want to check the recirculation of office AHUs we need the
    Return Air Temp, Mixed Air Temp, and Supply Air Temp. In the following
    case AHU01 is a lab AHU and does not have a Return Air Temp, so we don't
    need to analyze this AHU. If we filter that from our group of tags to
    pull data we would run the function as:

    group_tags(allTags, parentLevel=2, sensorGroup=['Mixed Air Temp',
                                                  'Return Air Temp',
                                                  'Supply Air Temp'])

     u'BLDG.AHU.AHU01.Supply Air Temp',
     u'BLDG.AHU.AHU01.Supply Air Temp Setpoint',

     u'BLDG.AHU.AHU02.Mixed Air Temp',
     u'BLDG.AHU.AHU02.Return Air Temp',
     u'BLDG.AHU.AHU02.Supply Air Temp',
     u'BLDG.AHU.AHU02.Supply Air Temp Setpoint',

    AHU01 is dropped because it does not have all 3 sensors, and AHU02
    Supply Air Temp Setpoint is also dropped because it is not part of the
    sensorGroup

     output = [u'BLDG.AHU.AHU02.Mixed Air Temp',
               u'BLDG.AHU.AHU02.Return Air Temp',
               u'BLDG.AHU.AHU02.Supply Air Temp',]

    NOTE: This function has nothing to do with missing data, or data quality


    Inputs:
        allTags (list or list of strings) - The pool of tags that may be pulled
        This list needs to be pregenerated by a function like search_tags()
        parentLevel (int) - The index of the
        sensorGroup (str or list of str) -

    """

    # Initial Vars
    ddict = collections.defaultdict(list)
    passValue = len(sensorGroup)
    wrongList, shortList = [], []

    # Group tags by splitting, and using the parentName as dict Key
    for tag in allTags:
        splitTag = tag.split('.')
        if splitTag[-1] in sensorGroup:
            ddict['.'.join(splitTag[0:parentLevel])].append(tag)

        else:
            wrongList.append(tag)

    # Check which parents have the proper # of children
    for parentKey in ddict:
        if len(ddict[parentKey]) < passValue:
            shortList.append(parentKey)

        else:
            pass
    # Delete Failures from defaultdict
    for key in shortList:
        del ddict[key]

    # Display which points were discarded and why
    print('Points with the wrong sensors')
    print(wrongList)
    print()
    print('Correct sensors, but not a complete set')
    print(shortList)

    # Re-combine items in dict into a flat list
    resultList = []
    for key, value in ddict.items():
        resultList += value

    return sorted(resultList)

# =============================================================================
# --- General Functions
# =============================================================================


def myprint(iterable):
    for i in range(len(iterable)):
        print('item:', i, iterable[i])


def pprint(iterable):
    for row in iterable:
        print(row)


def print_face():
    with open('face.txt', 'r') as f:
        for line in f:
            print(line.strip('\n'))


def read_lines(fileName):
    with open(fileName, 'r') as f:
        lines = f.readlines()
    return lines


def write_lines(newLines, newFile):
    with open(newFile, 'w') as f:
        for line in newLines:
            f.write(line)
        f.close()


def find_files(extension='.csv', filePath=None, relative=False):
    """
    This function returns a list of file names that have the _input_ extensions
    from a given input folder 'filePath'

    Inputs:
        filePath - string - full length path name to folder containing files

    Outputs:
        fileNames - list - list of strings where each string is the fullpath
        name to all of the .csv files located in 'filePath' input.
    """

    extLen = len(extension)

    # Use current directory that __main__ lives in
    if filePath is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
    else:
        dir_path = filePath

    if relative:
        preFix = os.path.dirname(os.path.realpath(__file__))
        dir_path = os.path.join(preFix, filePath)

    # get all files in dir
    dirList = os.listdir(dir_path)

    # Filter files that do not contain extension
    fileNames = []
    for item in dirList:
        if item[len(item) - extLen:len(item)] == extension:
            fileNames.append(item)

    return fileNames


def translate_name(inputName, langFrom='siemens', langTo='pi', regex=False):
    '''
    translate name is used to change the building names that are used in one
    system to the building names that are used in another system. This is
    needed when running an analysis that relies on comparing systems or data
    about the same building with other data related to that building. If they
    have different names then the program wont know what two names relate to
    the same building (until we get a nice metadata database to work with).

    Must call information from a manually maintained excel document

    can translate between SIEMENS PI CANN and WIFI data names

    inputs:
        inputName - str, list of str, pd.index of str
            These are the names to be translated. Translated names are returned
            in the object type that was supplied.

        langFrom - str - ['pi', 'siemens', 'wifi', 'CANN']

        langTo - str - ['pi', 'siemens', 'wifi', 'CANN']

        regex - bool - default = False

    '''
    '''
    1. cases:
        pandas columns with multiIndex
        pandas column with no mutliIndex
        single string
        list of strings

    2. open buildings.xlsx
    3. find building in langFrom, get CANN,
    4. Use CANN to find name in langTo


    5. return same type of object that was input
    '''

    # Single String
    if isinstance(inputName, str):

        mask = buildings[langFrom] == inputName

        if mask[mask].empty:
            # print("{} not found in langFrom: {}".format(inputName,langFrom))
            return None
        else:
            CAAN = mask[mask].index

        newName = buildings[langTo][CAAN][0]

        if isinstance(newName, float):
            return None
        else:
            return newName

    if isinstance(inputName, list):
        outList = []

        for inName in inputName:
            newName = translate_name(inName, langFrom=langFrom, langTo=langTo)
            outList.append(newName)

        return outList

    if isinstance(inputName, pd.core.indexes.base.Index):

        newColumns = []

        for i, name in enumerate(inputName):
            splitNames = name.split('.')
            firstName = splitNames[0]
            newName = translate_name(firstName,
                                     langFrom=langFrom,
                                     langTo=langTo)

            if isinstance(newName, float):
                newColumns.append(name)
            else:
                newColumns.append('.'.join([newName] + splitNames[1:]))

        return pd.DataFrame(columns=newColumns).columns


def siemens_data(filePath,
                 allFiles=False,
                 joinHeader=True,
                 twoColTime=True,
                 replacements=None,
                 makeMultiIndex=True,
                 skipRows=0,
                 convertCol=True,
                 ignoreLevel=None):
    """
    This function loads the Siemens reports that as a .csv __sam provides to
    me__. The function will first check the sheet for all of the column
    headings that are listed row by row before the data begins. It will extract
    these names and create a multi-index column set for the data point columns
    that are in the file.

    These reports have the time in two columns, but it is combined into one
    dateTime pandas index.

    If any replacements are to be made they are provided as a list of lists,
    where each internal list is the thing to be replaced and the replacement.

    columns are converted to numeric type if wanted

    Inputs:
        filePath - string - full length path to folder containing .csv files

        allFiles - bool - not implemented

        joinHeader - bool - in some siemens reports the points are listedd as
                    point_1, point_2 etc... and the points are given as a list
                    in the begining of the file. If joinHeader is TRUE, then it
                    will replace the column headings with the true point name,
                    then it will remove the first X rows that are no longer
                    needed before loading the data.

        twoColTime - bool - If the data is from siemens and has the standard
        two column time convert it to a single time index. If the Datetime is
        listed in column 1 then twocoltime should be set to FALSE.

        replacements - list of lists of strings - replaces the first value in
                    the inner most lists with the second value in the same
                    inner most list

        makeMultiIndex - bool - converts the standard PIpoint naming convention
                    inter a hierarchical multi index where spaces in the name
                    are first replaced with underscores (_), and then the name
                    is split at periods (.). The index will then group things
                    by AHU level, AHU number, AHU point data. This option can
                    be extremely useful for crunching large sets of data with
                    a farily homogenous naming style

        skipRows = integer - skip this many rows from the top of the input file

        convertCol - bool - If True, convert all columns to numeric type

        ignoreLevel - string - This allows the multiIndexing phase to remove a
                given string from the column headings so that there are not
                multiple levels in the index in which all data is contained.
                This was built out of necessity when it seemed that the
                multiIndex could not have 4 levels and thus we removed the top
                level since it was the building name, and all of the
                information was was for the same building, and thus all
                data was contained in the index rendering it superfluous

    Outputs:
        data - pd.DataFrame - dataframe containing all of the data from the
            loaded csv
    """
    # Join header
    if joinHeader is True:
        with open(filePath) as f:
            sensorList = []
            blankLine = 0
            for i, line in enumerate(f):

                print(line.find('<>Date'))

                if i > 1000:
                    errorMsg = 'Could not find <>Date for join header'
                    print(errorMsg)
                    return errorMsg
                if line.find('<>Date') >= 0:
                    blankLine = i
                    break
                else:
                    if i == 0:
                        continue
                    else:
                        sensorName = line.split(',')
                        sensorList.append(sensorName)
            sensorList = sensorList[0:len(sensorList) - 4]
            sensorList = [x[1].replace('"', '') for x in sensorList]
        f.close()
        data = pd.read_csv(filePath, skiprows=blankLine)
    else:
        data = pd.read_csv(filePath, skiprows=skipRows)

        if twoColTime is True:
            sensorList = data.columns[2:len(data.columns)]
        else:
            sensorList = data.columns[1:len(data.columns)]

    # Trim the fat
    data = data.iloc[0:len(data) - 1, :]  # report contains a useless last row

    # Two Column Time
    if twoColTime is True:
        timeIndex = data.iloc[:, 0] + ' ' + data.iloc[:, 1]
        data.index = pd.to_datetime(timeIndex)
        data = data.iloc[:, 2:len(data.columns)]
    else:
        timeIndex = data.iloc[:, 0]
        data.index = pd.to_datetime(timeIndex)
        data = data.iloc[:, 1:len(data.columns)]

    # Set sensor names as columns of df
    data.columns = sensorList

    # Replace data
    # Useful for replacing "OFF" as 0 and "ON" as 1 if needed
    if replacements is None:
        pass
    else:
        for item in replacements:
            replace_ = item[0]
            with_ = item[1]
            data = data.replace(replace_, with_)

    # Convert Cols to numeric
    if convertCol is True:
        for col in data.columns:
            if(data[col].dtype != np.number):
                data[col] = pd.to_numeric(data[col], errors="coerce")

    # Make multi-index. DEPRECATED
    if makeMultiIndex is True:
        print('MULTI INDEX has been removed from this function, '
              'change code to open data, and run mypy.make_mutli_index()'
              ' seperately')

    return data


def make_multi_index(columnIndex,
                     splitString='.',
                     ignoreLevel=None,
                     padLength=True,
                     keepSpace=True):
    """
    Takes a in a pd.column object and splits apart the name to create a
    hierarchical index based on the PI tag name notation: ACAD.AHU.AHU01.sensor

    The

    inputs:
        columnIndex - pd.columns - The items that will become the multiindex

        splitString - str - default = '.'
            This is the designator that will be used to split
            the strings in the tag name into different pieces.
            'ACAD.AHU.AHU01.sensor' -> ['ACAD', 'AHU', 'AHU01', 'sensor']

        ignoreLevel - str - default = None
            If there is a redundant level that can be removed from all tags
            specify here. Eg. 'ACAD' if all points are in ACAD

        padLength - bool - default = True
            If some tag names are not the same length as others it will pad
            empty strings on the end of the split sensor name ensuring that
            the short sensor names are on the lowest possible index level (0)

        keepSpace - bool - default = True
            If keepSpace = True all spaces in sensor name will become "_"

    returns:
        A multiindex object (pd.index)

    """

    # NOTE: I wrote this function a while ago and don't want to break it so i'm
    # leaving the code as somewhat ugly. but it works! - june 2018

    maxLength = 0
    columnList = []

    for sensor in columnIndex:
        #        sensor = sensor.replace('"','')
        if keepSpace:
            sensor = sensor.replace(' ', '~!~')

        # Split String
        if isinstance(splitString, list):
            for repString in splitString:
                sensor = sensor.replace(repString, ' ')
        elif isinstance(splitString, str):
            sensor = sensor.replace(splitString, ' ')

        if keepSpace:
            sensor = sensor.replace('~!~', '_')  # pd does not like spaces

        sensorList = sensor.split(' ')

        # Ignore Level
        if ignoreLevel:
            if isinstance(ignoreLevel, list):
                for level in ignoreLevel:
                    try:
                        sensorList.remove(level)
                    except ValueError:
                        print("{} not in {} for ignoreLevel".format(level,
                                                                    sensor))
                        pass

            elif isinstance(ignoreLevel, str):
                try:
                    sensorList.remove(ignoreLevel)
                except ValueError:
                    print("{} not in {} for ignoreLevel".format(ignoreLevel,
                                                                sensor))
                    pass

            else:
                print("Ignore level must be list of str or a str")

        if len(sensorList) > maxLength:
            maxLength = len(sensorList)

        columnList.append(sensorList)

    # Pad Length
    if padLength:
        for column in columnList:
            diff = maxLength - len(column)
            if diff > 0:
                for i in range(diff):
                    column.append('')

    columnList = list(map(list, zip(*columnList)))
    header = pd.MultiIndex.from_arrays(columnList)

    return header


def build_time_columns(dfIn, daytime=(8, 20)):
    """
    This function will add many columns to the end of the dataframe which
    contains information about the rows dateTime but split into these columns

    year, month, dayofmonth, hour, minute, dayofweek, weekday, saturday, sunday
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


def remove_time_cols(df):
    timeCols = ['year', 'month', 'dayofmonth', 'hour', 'minute', 'weekday',
                'daytime', 'dayofweek', 'saturday', 'sunday', 'weekofyear']

    return df.drop(timeCols, axis=1, errors='ignore')


def interval_checker(df, unit='min', silenced=True):
    """
    Calculate the intervalsize of a time series dataset and returns
    the mode() of the index timedeltas in specified unit eg: 60.0 min
    """

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

    if silenced:
        pass
    else:
        print('"{}" is {} {}'.format(timeDelta, output, unit))

    return output


def merge_oat(df, source='masterfile', choose=None):
    """
    Returns dataframe with additional OAT data column.

    """
    # Need datetime index to merge correctly
    if ~isinstance(df.index, pd.core.indexes.datetimes.DatetimeIndex):
        print('Must pass df with Datetime Index to merge_oat, returning df')
#        return df
    if isinstance(df, pd.DataFrame):
        print('its a df!')
        if df.empty:
            print('its empty too!')
            oatPath = path.join(path_prefix, 'OATmaster.csv')
            OAT = pd.read_csv(oatPath, index_col=0, parse_dates=True,
                              infer_datetime_format=True)
            return OAT

    # Deprecating
    if choose:
        print("choose options is beind deprecated due to vagueness")

    # Load data and pd.merge()
    if source == 'masterfile':
        oatPath = path.join(path_prefix, 'OATmaster.csv')
        print('Pulling data from: {}'.format(oatPath))

        OAT = pd.read_csv(oatPath, index_col=0, parse_dates=True,
                          infer_datetime_format=True)

        print('OAT master loaded!')
        df = pd.merge(df, OAT, right_index=True, left_index=True)
        print('Merge complete!')

    return df


def calculate_degree_days(oatData=None, by='day', cutoff=65):
    '''
    Calculate the HDD or CDD in a day or month as a standalone action
    Source of data is always OAT master file
    '''

    print("Deprecation warning: this function should be called "
          "calculate_degree_hours(). Use that name instead")

    if isinstance(oatData, pd.DataFrame) or isinstance(oatData, pd.Series):
        if isinstance(oatData, pd.Series):
            df = oatData.to_frame()
        else:
            df = oatData
        df.columns = ['OAT']
        print('OAT supplied with df')
    else:
        print('OAT being loaded from master file')
        oatPath = path.join(path_prefix, 'OATmaster.csv')

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

    # Resample again with sum() if needed
    if by.lower() == 'day' or by.lower() == 'd':
        hours = hours.resample('1D').sum()
    elif by.lower() == 'month' or by.lower() == 'm':
        hours = hours.resample('1M').sum()
    elif by.lower() == 'hour' or by.lower() == 'h':
        pass
    else:
        print("WARNING: calculate_degree_days didn't get - by=H, D or M")
    return hours


def build_degree_days(df, by='day', OAT='OAT', cutoff=65):

    print("Deprecation warning: this function should be called "
          "calculate_degree_hours(). Use that name instead")

    if OAT in df.columns:
        print('make_degree_days is using OAT passed in with df')
    else:
        df = merge_oat(df)

    df = df.resample('1H').mean()

    hours = pd.DataFrame()
    hours['HDH'] = cutoff - df[OAT]
    hours['CDH'] = df[OAT] - cutoff
    hours.index = df.index
    hours[hours < 0] = 0

    f = {}
    for col in df.columns:
        f[col] = 'mean'
    f['HDD'] = sum
    f['CDD'] = sum

    if by == 'day':

        hours = hours / 24

        hours.columns = ['HDD', 'CDD']
        df = pd.merge(df, hours, right_index=True, left_index=True)
        grouped = df.groupby(by=('year', 'month', 'day')).agg(f)

        return grouped

    elif by == 'hour':

        grouped = pd.merge(df, hours, right_index=True, left_index=True)

        return grouped


def calculate_degree_hours(oatData=None, by='day', cutoff=65):
    '''
    Calculate the HDH or CDH in a day or month as a standalone action
    Source of data is always OAT master file
    '''

    if isinstance(oatData, pd.DataFrame) or isinstance(oatData, pd.Series):
        if isinstance(oatData, pd.Series):
            df = oatData.to_frame()
        else:
            df = oatData
        df.columns = ['OAT']
        print('OAT supplied with df')
    else:
        print('OAT being loaded from master file')
        oatPath = path.join(path_prefix, 'OATmaster.csv')

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

    # Resample again with sum() if needed
    if by.lower() == 'day' or by.lower() == 'd':
        hours = hours.resample('1D').sum()
    elif by.lower() == 'month' or by.lower() == 'm':
        hours = hours.resample('1M').sum()
    elif by.lower() == 'hour' or by.lower() == 'h':
        pass
    else:
        print("WARNING: calculate_degree_days didn't get by=H, D or M")
    return hours


def build_degree_hours(df, by='day', OAT='OAT', cutoff=65):
    """
    Add columns to a dataframe containing the HDH and CDH numbers, resampled
    to the sampling rate specified by "by="
    """

    if OAT in df.columns:
        print('make_degree_days is using OAT passed in with df')
    else:
        df = merge_oat(df)

    df = df.resample('1H').mean()

    hours = pd.DataFrame()
    hours['HDH'] = cutoff - df[OAT]
    hours['CDH'] = df[OAT] - cutoff
    hours.index = df.index
    hours[hours < 0] = 0

    f = {}
    for col in df.columns:
        f[col] = 'mean'
    f['HDD'] = sum
    f['CDD'] = sum

    hours = pd.merge(df, hours, right_index=True, left_index=True)

    if by.lower() == 'day' or by.lower() == 'd':
        hours = hours.resample('1D').agg(f)
    elif by.lower() == 'month' or by.lower() == 'm':
        hours = hours.resample('1M').agg(f)
    elif by.lower() == 'hour' or by.lower() == 'h':
        pass
    else:
        print("WARNING: calculate_degree_days didn't get -- by=H, D or M")

    if hours.isnull().any().any():
        print('some HDH/CDH detected as null in build_degree_hours')
        print('printing isnull().value_counts()')
        print(hours.isnull.value_counts())

        hours = hours.dropna(axis=1, how='any')

    return hours


def pivot_and_plot(df,
                   split=[],
                   stack=[],
                   xaxis=['hour'],
                   aggFun='mean',
                   plotType='bar',
                   splitSensors=False,
                   transpose=False,
                   wrapNames=False,
                   customColors='tab10',
                   titleTag="",
                   ylabel=None,
                   figSize=(6, 5),
                   ylims=None,
                   saveFigs=False,
                   interpTime=True):

    dayReplacements = {0: 'Monday',
                       1: 'Tuesday',
                       2: 'Wednesday',
                       3: 'Thursday',
                       4: 'Friday',
                       5: 'Saturday',
                       6: 'Sunday'}

    monthReplacements = {1: 'Jan',
                         2: 'Feb',
                         3: 'Mar',
                         4: 'Apr',
                         5: 'May',
                         6: 'Jun',
                         7: 'Jul',
                         8: 'Aug',
                         9: 'Sep',
                         10: 'Oct',
                         11: 'Nov',
                         12: 'Dec'}

    # make a plot for each combination of split
    # if pivoting on sensors, make sure time columns are not mixed up
    columnNames = list(df.columns)
    df = build_time_columns(df)

    if interpTime:
        for time in interpTime:
            if time == 'dayofweek':
                df['dayofweek'] = df['dayofweek'].replace(dayReplacements)
            if time == 'month':
                df['month'] = df['month'].replace(monthReplacements)

            if time == 'daytime':
                df['daytime'] = df['daytime'].replace({0: 'night', 1: 'day'})

    _values = columnNames

    # Make Pivot table where split and stack are defined
    pivoted = pd.pivot_table(df, values=_values,
                             index=split + xaxis,
                             columns=stack,
                             aggfunc=aggFun)

    # Rearrange sensor, sort
    if splitSensors:
        split = ['sensor'] + split
        pivoted = pivoted.stack(level=0)
        nblevels = pivoted.index.nlevels
        pivoted.index.set_names('sensor', level=nblevels - 1, inplace=True)

        # switch index
        order = [nblevels - 1] + list(range(0, nblevels - 1))
        pivoted = pivoted.reorder_levels(order).sort_index()

    # for tup list, make figures or for level values make figures
    valueLists = []
    for i, level in enumerate(split):
        #        print(pivoted.index.get_level_values(i).unique())
        valueLists.append(list(pivoted.index.get_level_values(i).unique()))
    splitTuples = list(itertools.product(*valueLists))
    splitLevels = [split.index(x) for x in split]

    for tup in splitTuples:

        if tup:
            splitData = pivoted.xs(tup, level=splitLevels)
        else:
            splitData = pivoted

        if transpose:
            splitData = splitData.T

        # re-order time interp columns
        if 'dayofweek' in interpTime:
            splitData = splitData[['Monday', 'Tuesday', 'Wednesday',
                                   'Thursday', 'Friday', 'Saturday', 'Sunday']]
            pass

        # Color section
        if isinstance(customColors, str):
            if customColors == 'gradient':
                # figure out a gradient
                colors = 'viridis'
            else:
                colors = customColors

        elif isinstance(customColors, list):
            colors = customColors

        # Barplot
        if plotType == 'bar':
            ax = splitData.plot(kind='bar', cmap=colors, figsize=figSize)

#            for p in ax.patches:
#                ax.annotate(str(int(p.get_height())),
#                           (p.get_x() * 1.005, p.get_height() * 1.005))

            wrapLen = 15
            if wrapNames:
                labels = ['\n'.join(wrap(l, wrapLen)) for l in
                          list(splitData.index)]
                ax.set_xticklabels(labels)

        # Lineplot
        if plotType == 'line':
#            print(splitData)
            ax = splitData.plot(kind='line', cmap=colors, figsize=figSize)

        titleString = str(list(zip(split, tup)))
        plt.title(titleString + " " + titleTag)
        plt.ylim(ylims)

        if transpose:
            plt.legend(bbox_to_anchor=[1, 0],
                       title=df.columns.name,
                       loc='lower left')
        else:
            plt.legend(bbox_to_anchor=[1, 0],
                       title=stack,
                       loc='lower left')

        plt.tight_layout()

        if ylabel:
            plt.ylabel(ylabel)

        if saveFigs:
            plt.savefig('{}.png'.format(titleString).replace(':', '_'),
                        bbox_inches='tight')

        plt.show()

    return pivoted


def scatter_grid(df, xaxis='index', figsize=(20, 20), xbounds=None, alpha=0.1):
    size = len(df.columns)

    cols = df.columns

    col = round(size ** 0.5 + 0.49999)
    row = round(size / col)
    if row * col <= size:  # make <= so there is an empty last corner for scale
        row += 1

    if xaxis == 'index':
        xdata = df.index
    else:
        xdata = df[xaxis]

#    deadCells = row * col - size
    fig, axes = plt.subplots(col, row, figsize=figsize,
                             sharex='row', sharey='col')

    plotIndex = -1
    for c in range(col):
        for r in range(row):
            ax = axes[c, r]
            plotIndex += 1

            if plotIndex >= size:
                ax.scatter(0, 0, s=1, color='w')

            else:
                try:
                    ax.scatter(xdata, df.iloc[:, plotIndex],
                               s=1, color='k', alpha=alpha)
                    plt.xlim(xbounds)

                except IndexError:
                    print('No data in dataset {}'.format(plotIndex))

            if plotIndex < (row * col) - 1:
                ax.axis('off')
            if plotIndex < size:
                ax.set_title(cols[plotIndex], fontsize=6)

    plt.tight_layout(pad=-1.5)
    plt.show()


# ==============================================================================
# --- Retired Functions
# ==============================================================================

def find_string(string, startString='AHU', endString='.'):
    startIndex = string.find(startString)
    endIndex = string[startIndex + 1:len(string)].find(endString)

    if startIndex < 0 or endIndex < 0:
        print(startString, ' string missing')
        return ''
    return string[startIndex:endIndex + startIndex + 1]


def average_by_day(data,
                   resampleInterval='30T',
                   by=None,
                   timeColumnsExist=False):

    assert(isinstance(data.index, pd.DatetimeIndex))

    if not by:
        by = ('saturday', 'sunday', 'hour', 'minute')

    if timeColumnsExist:
        pass
    else:
        data = build_time_columns(data)

    data = data.resample(resampleInterval).mean()
    grouped = data.groupby(by=by).mean()

    return grouped


def group_slice(df,
                by=('day'),
                splitLevels=None,
                aggFun='sum',
                unstackSensors=False):
    """
    The aim of this function is to create a collection of dfs that are sliced
    on a variety of time based categories (reliant on mypy.build_time_columns).
    The dfs are sliced with df.xs() so a copy is generate
    some time axes can be ignored, and so they will live together on the index
    for future plotting needs

    inputs:
        df - pd.DataFrame - A timeseries dataframe with categorical time cols
        by - tuple of strings - The groupby inputs (must match column names)
        levels - list of ints - Levels that will be xs()'d
        aggFun - str - usually 'sum' or 'mean' depending on the data
    """
    # Group levels
    grouped = df.groupby(by=by).agg(aggFun)
    grouped = remove_time_cols(grouped)

    if unstackSensors:
        grouped = grouped.stack()
        nblevels = grouped.index.nlevels
        order = [nblevels - 1] + list(range(0, nblevels - 1))
        grouped = grouped.reorder_levels(order).sort_index()

    # Find all unique groupings possible
    valueLists = []

    if splitLevels:
        for level in splitLevels:
            valueLists.append(list(
                    grouped.index.get_level_values(level).unique()))
        listOfTuples = list(itertools.product(*valueLists))
    else:

        print('What else would this function do without splitLevels?')
        raise ValueError('SplitLevels must be list of ints corresponding to'
                         'the "by" inputs')
        return

    # Construct Dataframe dictionary
    dfs = {}

#    print(listOfTuples)
    for tup in listOfTuples:
        # TODO: Make naming better (use splitLevels? use name dict?)
        section = grouped.xs(tup, level=splitLevels)
        print(section)

        name = ''
#
#        if unstackSensors == True:
#            for i, lev in enumerate(splitLevels):
#                name += str(tup[i])
#        else:

        for i, lev in enumerate(splitLevels):
            name += str(by[lev]) + ': ' + str(tup[i])
            name += '_'
        name = name.strip('_')
        name += '_' + str(aggFun)

        if ~section.empty:
            dfs[name] = section

    return dfs


def plot_group_slice(dfs,
                     transpose=False,
                     kind='bar',
                     layer=None,
                     unstackSensors=False,
                     saveFig=False,
                     wrapNames=False,
                     wrapLen=15,
                     tag=''):

    for tup, df in dfs.items():
        #        df = remove_time_cols(df)

        if df.empty:
            continue

        if transpose:
            df = df.T

        if kind == 'bar':
            ax = df.plot(kind='bar')
            if wrapNames:
                labels = ['\n'.join(wrap(l, wrapLen)) for l in list(df.index)]
                ax.set_xticklabels(labels)

        if kind == 'line':

            if unstackSensors:
                unique = df.index.get_level_values(layer).unique()
                for u in unique:
                    sl = df.xs(u, level=layer)
                    sl.plot(label=u)
            elif layer:
                pass
            else:
                df.plot(kind='line')

        plt.title(tup)

        if transpose:
            plt.legend(bbox_to_anchor=[1, 0], title=df.columns.name,
                       loc='lower left')
        else:
            plt.legend(bbox_to_anchor=[1, 0], loc='lower left')
        plt.tight_layout()

        if saveFig:
            plt.savefig(tag + '{}.png'.format(tup).replace(':', '_'))
        plt.show()


def slice_and_plot(df,
                   by=('day'),
                   splitLevels=None,
                   aggFun='sum',
                   unstackSensors=False,

                   transpose=False,
                   kind='bar',
                   layer=None,
                   saveFig=False,
                   wrapNames=False,
                   wrapLen=15,
                   tag=''):

    if unstackSensors:
        assert(len(splitLevels) == 1)

    dfs = group_slice(df, by=by,
                      splitLevels=splitLevels,
                      aggFun=aggFun, unstackSensors=unstackSensors)

    print(len(dfs), 'THIS MANY DFs')

    plot_group_slice(dfs, transpose=transpose, wrapNames=wrapNames,
                     kind=kind, unstackSensors=unstackSensors, layer=layer,
                     saveFig=saveFig, wrapLen=wrapLen, tag=tag)


def standardize_columns(sensors):

    replaces = {
        'nil': 'nil',
        'SAS': 'Supply_Air_Temp_Sp',
        'Supply_Air_Temp_Sp': 'Supply_Air_Temp_Sp',

        'SAT': 'Supply_Air_Temp',
        'Supply_Air_Temp': 'Supply_Air_Temp',

        'HCO': 'Heating_Valve_Output',
        'HCV': 'Heating_Valve_Output',
        'Heating_Valve_Output': 'Heating_Valve_Output',

        'Supply_Air_Fan_Start/Stop': 'Supply_Air_Fan_Start/Stop',

        'HWS': 'Building_Hot_Water_Supply_Temp',
        'Building_Hot_Water_Supply_Temp': 'Building_Hot_Water_Supply_Temp'}

    newSensor = []
    for i, heading in enumerate(sensors):
        rep = None
        try:
            rep = replaces[heading]
            if rep:
                newSensor.append(rep)
        except:
            newSensor.append(heading)
            # print('ValueError: ', heading, 'not in standard headings')
            # print('Check the replaces varaible in standardize_columns')

    return newSensor


def find_csv(extension='.csv', filePath=None):
    """
    This function returns a list of file names that have the .csv extensions
    from a given input folder 'filePath'

    Inputs:
        filePath - string - full path name to folder containing .csv files

    Outputs:
        fileNames - list - list of strings where each string is the fullpath
        name to all of the .csv files located in 'filePath' input.
    """
    print('FIND_CSV IS BEING REMOVED, CHANGE TO '
          'find_files(filePath,extension=".csv")')

    extLen = len(extension)

    if filePath is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
    else:
        dir_path = filePath

    os.chdir(dir_path)
    dirList = os.listdir()

    fileNames = []
    for item in dirList:
        if item[len(item) - extLen:len(item)] == extension:
            fileNames.append(item)
    return fileNames


# ==============================================================================
# --- TESTS
# ==============================================================================


def test_translate_names1():

    to = 'wifi'
    fro = 'siemens'

    # string Test
    stringTest = 'Ghausi'
    print('StringTest')
    print(translate_name(stringTest, langFrom=fro, langTo=to))
    print('expected output for siemens -> wifi')
    print('GHAUSI-HALL')
    print('')

    # List Test
    listTest = ['Ghausi', 'PES', 'MRAK', 'SLAB']
    print('ListTest')
    print(translate_name(listTest, langFrom=fro, langTo=to))
    print('expected output for siemens -> wifi')
    print("['GHAUSI-HALL', 'PES', 'MRAK', 'SCIENCELAB']")

    # pd columns Test
    columnTest = pd.DataFrame(columns=listTest)
    columns = columnTest.columns
    print('ColumnTest')
    print(translate_name(columns, langFrom=fro, langTo=to))
    print('expected output for siemens -> wifi')
    print("Index(['GHAUSI-HALL', 'PES', 'MRAK', 'SCIENCELAB'],dtype='object')")

    # pd long columns Test
    longColumnTest = pd.DataFrame(columns=['Ghausi.AHU.AHU01.temp',
                                           'PES.noChange',
                                           'MRAK.noChange',
                                           'SLAB.ZONE.AHU01.RMT'])

    longColumns = longColumnTest.columns
    print('ColumnTest')
    print(translate_name(longColumns, langFrom=fro, langTo=to))
    print('expected output for siemens -> wifi')
    print('')


def test_translate_names2():
    fileName = "test_wifi_translate.csv"
    data = pd.read_csv(fileName, index_col=0, header=0)
    translated = translate_name(data.columns, langFrom='wifi', langTo='pi')

    return data.columns, translated


def test_calculate_degree_days():
    # Functions handles differeenct cases by= using by.lower()
    df1 = calculate_degree_days(by='D')
    df2 = calculate_degree_days(by='MONTH')
    df3 = calculate_degree_days(by='Hour')

    print(df1)
    print(df2)
    print(df3)


def repickle_test():

    picklePath = r'N:\ACE\Ckoshnick\git zone\mypy\testpickle.pk1'
    piTags = ['ACAD.AHU.AHU01.Supply Air Temp']
    csvOut = True
    rp = repickler(picklePath, piTags, csvOut)

    return rp


if __name__ == '__main__':
    ts = sample_data.time_series()
    ints = sample_data.ints()
    rand = sample_data.rand(size=(300, 20))
    scatter_grid(rand, xaxis='b',
                 figsize=(4, 4))

#    test_translate_names1()
#    D, E = test_translate_names2()
#    rp = repickle_test()
#    test_calculate_degree_days()

#    B = pivot_and_plot(ts)

#    B = pivot_and_plot(ts,
#                   split=['weekday'],
#                   stack=[],
#                   xaxis=['month'],
#                   aggFun='mean',
#                   plotType = 'bar',
#                   customColors='plasma',
#                   transpose=True,
#                   splitSensors=False,
#                   figSize=(10,6),
#                   ylims=(0,75),
#                   saveFigs=True)

#    import cProfile
#    cProfile.run('calculate_degree_days()')

    pass
