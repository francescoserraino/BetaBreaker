#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 02:46:48 2020

@author: francescoserraino
"""

"""
Created on Wed Dec  9 23:03:23 2020

@author: francescoserraino
"""
import re
import pandas as pd
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import scipy.spatial.distance as dist
import math
from sklearn.model_selection import train_test_split
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
import sys
sys.path.append("/".join(x for x in __file__.split("/")[:-1]))
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, NoTransition, CardTransition
from kivy.properties import DictProperty
from kivy.uix.button import ButtonBehavior
from kivy.uix.image import Image
from kivy.uix.scrollview import ScrollView
import sys
sys.path.append("/".join(x for x in __file__.split("/")[:-1]))
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, NoTransition, CardTransition
from kivy.properties import DictProperty

from functools import partial
from os import walk
from datetime import datetime
import kivy.utils
from kivy.utils import platform
import requests
import json
import traceback
from kivy.graphics import Color, RoundedRectangle
import pickle
from itertools import chain, combinations
import requests
from kivy.base import runTouchApp
from kivy.lang import Builder
from kivy.properties import NumericProperty
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.lang import Builder
from kivy.properties import StringProperty
from random import randint




MY_GLOBAL = "test"



class Functions():
    

    
    model = pickle.load(open('finalized_model.sav', 'rb'))
    
    
    
    def grade_predictor(self, start_holds, intermediate_holds, finish_holds):
        data = pd.read_csv('new_data_clean.csv')
        data.set_index('name', inplace = True)
        
        model = pickle.load(open('finalized_model.sav', 'rb'))    
        hold_class_dict = {
        'a1':0,'a2':0,'a3':0,'a4':0,'a5':3,'a6':0,'a7':0,'a8':0,'a9':3,'a10':1,'a11':1,'a12':2,'a13':1,'a14':2,'a15':1,'a16':1,'a17':0,'a18':3,
        'b1':0,'b2':0,'b3':3,'b4':1,'b5':0,'b6':1,'b7':1,'b8':2,'b9':3,'b10':1,'b11':2,'b12':1,'b13':2,'b14':0,'b15':1,'b16':3,'b17':0,'b18':2,
        'c1':0,'c2':0,'c3':0,'c4':0,'c5':3,'c6':2,'c7':2,'c8':3,'c9':1,'c10':1,'c11':2,'c12':1,'c13':3,'c14':1,'c15':1,'c16':2,'c17':0,'c18':1,
        'd1':0,'d2':0,'d3':1,'d4':0,'d5':2,'d6':2,'d7':3,'d8':1,'d9':1,'d10':1,'d11':2,'d12':2,'d13':1,'d14':2,'d15':2,'d16':1,'d17':2,'d18':3,
        'e1':0,'e2':0,'e3':0,'e4':0,'e5':0,'e6':3,'e7':2,'e8':3,'e9':2,'e10':2,'e11':2,'e12':2,'e13':2,'e14':2,'e15':2,'e16':1,'e17':0,'e18':3,
        'f1':0,'f2':0,'f3':0,'f4':0,'f5':3,'f6':1,'f7':1,'f8':1,'f9':2,'f10':2,'f11':2,'f12':2,'f13':2,'f14':2,'f15':2,'f16':2,'f17':0,'f18':0,
        'g1':0,'g2':2,'g3':0,'g4':3,'g5':0,'g6':3,'g7':1,'g8':3,'g9':2,'g10':2,'g11':1,'g12':2,'g13':3,'g14':3,'g15':2,'g16':1,'g17':2,'g18':2,
        'h1':0,'h2':0,'h3':0,'h4':0,'h5':2,'h6':0,'h7':1,'h8':3,'h9':2,'h10':3,'h11':3,'h12':3,'h13':2,'h14':2,'h15':1,'h16':1,'h17':0,'h18':1,
        'i1':0,'i2':0,'i3':0,'i4':3,'i5':3,'i6':2,'i7':2,'i8':1,'i9':1,'i10':3,'i11':2,'i12':2,'i13':2,'i14':2,'i15':2,'i16':2,'i17':0,'i18':2,
        'j1':0,'j2':1,'j3':0,'j4':0,'j5':3,'j6':1,'j7':2,'j8':3,'j9':2,'j10':2,'j11':1,'j12':2,'j13':2,'j14':1,'j15':0,'j16':2,'j17':0,'j18':0,
        'k1':0,'k2':0,'k3':0,'k4':0,'k5':3,'k6':1,'k7':1,'k8':1,'k9':2,'k10':1,'k11':2,'k12':1,'k13':1,'k14':2,'k15':0,'k16':2,'k17':0,'k18':3,                
        }
        
        grade_dict = {'4+':-1,'5':0,'5+':1,'6A':2,'6A+':3,'6B':4,'6B+':5,'6C':6,'6C+':7,'7A':8,'7A+':9,'7B':10,'7B+':11,'7C':12,'7C+':13,'8A':14,'8A+':15,'8B':16, '8B+':17}
        
        def all_subsets(ss):
            return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))
    
    
        def frac_hard_2016(row, holds):
            count = 0
            for item in hold_class_dict.keys():
                if hold_class_dict[item] == 1:
                    if item in holds:
                        count +=1
                    else:
                        pass
                else:
                    pass
            return count/len(holds)
        
        def frac_medium_2016(row, holds):
            count = 0
            for item in hold_class_dict.keys():
                if hold_class_dict[item] == 2:
                    if item in holds:
                        count +=1
                    else:
                        pass
                else:
                    pass
            return count/len(holds)
        
        def frac_easy_2016(row, holds):
            count = 0
            for item in hold_class_dict.keys():
                if hold_class_dict[item] == 3:
                    if item in holds:
                        count +=1
                    else:
                        pass
                else:
                    pass
            return count/len(holds)
        
        def max_cons_dist_finder(row):
            distances = []
            for i in range(len(row)-1):
                first = row[i]
                second = row[i+1]
                distances.append(np.sqrt((float(first[0])-float(second[0]))**2+(float(first[1])-float(second[1]))**2))
            return np.max(distances)
        
        def avg_cons_dist_finder(row):
            distances = []
            for i in range(len(row)-1):
                first = row[i]
                second = row[i+1]
                distances.append(np.sqrt((float(first[0])-float(second[0]))**2+(float(first[1])-float(second[1]))**2))
            return np.mean(distances)
        
        def average_dist_finder(row):
            distances = []
            row_1 = row
            row_2 = row
            for x,y in row_1:
                for a,b in row_2:
                    distances.append(np.sqrt((float(x)-float(a))**2+(float(y)-float(b))**2))
            return np.mean(distances)
        
        def coords_to_num(the_list):
            dict_to_loc = {'a':1,'b':2,'c':3,'d':4,'e':5,'f':6,'g':7,'h':8,'i':9,'j':10, 'k':11}   
            new_list = []
            for item in the_list:
                y = item[1:]
                x = dict_to_loc[item[0]]
                new_list.append((x,y))
            return new_list
            
        def get_key(val): 
                for key, value in grade_dict.items(): 
                     if val == value: 
                        return key 
            
        start = re.findall('[a-k][0-9]+', start_holds)
        middle =  re.findall('[a-k][0-9]+', intermediate_holds)
        end = re.findall('[a-k][0-9]+',finish_holds)
        holds = start+middle+end
        for item in holds:
            if item not in hold_class_dict.keys():
                return 'Please enter a valid climb\nYou entered invalid holds'
            elif hold_class_dict[item] == 0:
                return 'Please enter a valid climb\nYou entered invalid holds'
            else:
                pass
        if len(holds)<=3:
            return 'Please enter a valid climb\nYou need at least 4 holds'
        elif int(holds[-1][1:]) != 18:
            return 'Please enter a valid climb\nYou need at least one finish hold'
        else:
            
            num_list = []
            a = all_subsets(holds)
            the_subsets = list(a)
            subsets = [item for item in the_subsets if len(item)>3]
            for item in subsets:
                coords = coords_to_num(holds)
                row_list = []       
                for item in data.reset_index().drop(['grade_num','easy','name', 'average_move_distance'], axis = 1).columns:
                    if item in holds:
                        row_list.append(1)
                    elif item == 'no_of_moves':
                        row_list.append(sum(row_list))
                    elif item == 'average_seperation':
                        row_list.append(average_dist_finder(coords))
                    elif item == 'average_move_distane':
                        row_list.append(avg_cons_dist_finder(coords))
                    elif item == 'max_move_distance':
                        row_list.append(max_cons_dist_finder(coords))
                    elif item == 'easy':
                        row_list.append(frac_easy_2016(row_list, holds))
                    elif item == 'medium':
                        row_list.append(frac_medium_2016(row_list, holds))
                    elif item == 'hard':
                        row_list.append(frac_hard_2016(row_list, holds))
                    else:
                        row_list.append(0)
                row = np.array([row_list])
        
                num_list.append(model.predict(row))
            num = min(num_list)
            grade_suggest = get_key(num)
            if item == 1:
                grade_suggest_upper = '9a'
            else:
                grade_suggest_upper = get_key(num+1)
            if item == -1:
                grade_suggest_lower = '4'
            else:
                grade_suggest_lower = get_key(num-1)
            grade_suggest = get_key(num)
            
            return f'Suggested Grade: {grade_suggest}\nSuggested Range: {grade_suggest_lower} - {grade_suggest_upper}'

    def closest_vid_finder(self, start_holds, intermediate_holds ,end_holds):
        hold_class_dict = {
        'a1':0,'a2':0,'a3':0,'a4':0,'a5':3,'a6':0,'a7':0,'a8':0,'a9':3,'a10':1,'a11':1,'a12':2,'a13':1,'a14':2,'a15':1,'a16':1,'a17':0,'a18':3,
        'b1':0,'b2':0,'b3':3,'b4':1,'b5':0,'b6':1,'b7':1,'b8':2,'b9':3,'b10':1,'b11':2,'b12':1,'b13':2,'b14':0,'b15':1,'b16':3,'b17':0,'b18':2,
        'c1':0,'c2':0,'c3':0,'c4':0,'c5':3,'c6':2,'c7':2,'c8':3,'c9':1,'c10':1,'c11':2,'c12':1,'c13':3,'c14':1,'c15':1,'c16':2,'c17':0,'c18':1,
        'd1':0,'d2':0,'d3':1,'d4':0,'d5':2,'d6':2,'d7':3,'d8':1,'d9':1,'d10':1,'d11':2,'d12':2,'d13':1,'d14':2,'d15':2,'d16':1,'d17':2,'d18':3,
        'e1':0,'e2':0,'e3':0,'e4':0,'e5':0,'e6':3,'e7':2,'e8':3,'e9':2,'e10':2,'e11':2,'e12':2,'e13':2,'e14':2,'e15':2,'e16':1,'e17':0,'e18':3,
        'f1':0,'f2':0,'f3':0,'f4':0,'f5':3,'f6':1,'f7':1,'f8':1,'f9':2,'f10':2,'f11':2,'f12':2,'f13':2,'f14':2,'f15':2,'f16':2,'f17':0,'f18':0,
        'g1':0,'g2':2,'g3':0,'g4':3,'g5':0,'g6':3,'g7':1,'g8':3,'g9':2,'g10':2,'g11':1,'g12':2,'g13':3,'g14':3,'g15':2,'g16':1,'g17':2,'g18':2,
        'h1':0,'h2':0,'h3':0,'h4':0,'h5':2,'h6':0,'h7':1,'h8':3,'h9':2,'h10':3,'h11':3,'h12':3,'h13':2,'h14':2,'h15':1,'h16':1,'h17':0,'h18':1,
        'i1':0,'i2':0,'i3':0,'i4':3,'i5':3,'i6':2,'i7':2,'i8':1,'i9':1,'i10':3,'i11':2,'i12':2,'i13':2,'i14':2,'i15':2,'i16':2,'i17':0,'i18':2,
        'j1':0,'j2':1,'j3':0,'j4':0,'j5':3,'j6':1,'j7':2,'j8':3,'j9':2,'j10':2,'j11':1,'j12':2,'j13':2,'j14':1,'j15':0,'j16':2,'j17':0,'j18':0,
        'k1':0,'k2':0,'k3':0,'k4':0,'k5':3,'k6':1,'k7':1,'k8':1,'k9':2,'k10':1,'k11':2,'k12':1,'k13':1,'k14':2,'k15':0,'k16':2,'k17':0,'k18':3,                
        }
        data_vids = pd.read_csv('videos_2016_only.csv')
        start = re.findall('[a-k][0-9]+', start_holds)
        middle =  re.findall('[a-k][0-9]+', intermediate_holds)
        end = re.findall('[a-k][0-9]+',end_holds)
        holds = start+middle+end
        for item in holds:
            if item not in hold_class_dict.keys():
                return 'Please enter a valid climb\nYou entered invalid holds'
            elif hold_class_dict[item] == 0:
                return 'Please enter a valid climb\nYou entered invalid holds'
            else:
                pass
        if len(holds)<=3:
            return 'Please enter a valid climb\nYou need at least 4 holds'
        elif int(holds[-1][1:]) != 18:
            return 'Please enter a valid climb\nYou need at least one finish hold'
        else:
            data_vids_here = data_vids.drop(columns = ['name','grade_num','no_of_moves','average_move_distance','max_move_distance','easy','medium','hard'], axis = 1)
            row = []
            for item in data_vids_here.drop('video_link', axis = 1).columns:
                if item in holds:
                    row.append(1)
                else:
                    row.append(0)
            board_new = data_vids_here.drop('video_link', axis = 1)
            board_new.loc['new'] = row
            square = dist.squareform(dist.pdist(board_new))[-1][:-1]
            index = np.where(square == np.amin(square))[0][0]
            return data_vids_here.iloc[index]['video_link']






