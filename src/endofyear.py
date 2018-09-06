# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 08:48:48 2018

@author: koshnick
"""

#import mnv14 as mnv
import os
import nbformat

from nbconvert.preprocessors import ExecutePreprocessor

"""
This file will populate a list of folders that all contain the same file
structure. Inside of this sturcture will be an ipynb that will have all of the
important information of the model prefilled in (or read to be filled in from
file). The purpose of this script is to run all of those notebooks with a
keyword argument for the "FINAL DATE" and "FINAL OUTPUT (?)" so the output
function in the notebook will drop all of the important MnV information in a
central folder in the FY folder


Find folders - func
verify integrity?

put notebook in nbconvert
 - remove create archive line?
 - append final archive line?
 -- lets prototype with current archive line - no alterations

Run notebook - drop some output file in central folder (archive input?)
Can the notebook return data into native python namespace?

"""
directory = 'FY 2018-2019'


def find_folders(directory='FY 2018-2019'):
    folders = next(os.walk(directory))[1]
    removals = ['REPORT', '.ipynb_checkpoints']

    for folder in folders:
        if folder in removals:
            folders.remove(folder)

    return folders


def load_notebook(fileName):

    nb = nbformat.read(open(fileName), as_version=4)

    return nb


def edit_lines(nb):


    lastCell = nb['cells'].pop()
    print(lastCell)
    # 'mnv.create_archive(dk, mc, saveFigs=True)
    code = lastCell['source']
    code = code.replace(')', ', centralize=True)')

    lastCell['source'] = code

    nb['cells'].append(lastCell)

    return nb


def run_notebook(ep, nb):
    ep.preprocess(nb, {'metadata': {'path': None}})


    print('process finished')


def main():
    folders = find_folders(directory=directory)

    fileName = 're-Model-MnV.ipynb'
    ep = ExecutePreprocessor(timeout=6000, kernel_name='python3')

    for folder in folders:
        print(folder)

        os.chdir(os.path.join(directory,folder))

        nb = load_notebook(fileName)
        nb = edit_lines(nb)
        run_notebook(ep, nb)

        os.chdir(os.path.join('..','..'))


if __name__ == "__main__":
    A =  main()
