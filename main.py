
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
from functions import Functions
from itertools import chain, combinations
from kivy.app import App
from kivy.uix.popup import Popup
from kivy.lang import Builder
from kivy.uix.button import Button
import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.core.text import Label as CoreLabel
from kivy.uix.label import Label
from kivy.base import runTouchApp
from kivy.lang import Builder
from kivy.properties import NumericProperty
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.lang import Builder
from kivy.properties import StringProperty
from random import randint
import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
import requests
from kivy.properties import StringProperty
import time
from small_func import test_func



class P(FloatLayout):
#    def call_value(self):
#        x = Functions.grade_predictor(self,'a1', 'd4,d5', 'd3,g6')
#    pass
    idea_texto = StringProperty() 
        # <<<<<<<<<<<<<<<<<<<<
    def __init__(self, idea, **kwargs):
        super(P, self).__init__(**kwargs)
        self.idea_texto = idea


#class Widgets(Widget):
#    def btn(self):
#        show_popup()

def show_popup():
#    my_list = moonboardscreen.get_text_inputs(moonboardscreen)
#    start= my_list[0]
#    inter = my_list[1]
#    finish = my_list[2]
    with open('surname.txt') as f:
        idea = f.read()#Functions.grade_predictor(Functions, 'a2', 'd4,d5', 'd3,g6')
    show = P(idea) # Create a new instance of the P class 

    popupWindow = Popup(title='RESULTS', content=show, size_hint=(None,None),size=(300,300)) 
    # Create the popup window

    popupWindow.open()
    
class HomeScreen(Screen):
    pass
class moonboardscreen(Screen):

    def btn(self):
        show_popup()
#    def load(self):
#        with open("surname.txt") as fobj:
#            for surname in fobj:
#                self.surname = surname.rstrip()

class results(Screen):
    pass
class ImageButton(ButtonBehavior, Image):
    pass


GUI = Builder.load_file('main.kv')

class MainApp(App):

    def build(self):
        self.functions = Functions()
        self.small_func = test_func()
        return GUI
    
    def change_screen(self, screen_name):
        screen_manager = self.root.ids['screen_manager']
        screen_manager.current = screen_name

        
        
        
        
        
        
MainApp().run()