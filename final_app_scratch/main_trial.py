#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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


class HomeScreen(Screen):
    pass
class InsertData(Screen):
    def check_mail(self):
        addressToVerify = self.ids.email_main.text
        match = re.match('^[_a-z0-9-]+(\.[_a-z0-9-]+)*@[a-z0-9-]+(\.[a-z0-9-]+)*(\.[a-z]{2,4})$', addressToVerify)
        if match == None:
            print('Email not valid!')
        else:
            MainApp().change_screen('home_screen')
GUI = Builder.load_file("insertdata.kv")
class MainApp(App):
    def build(self):
        return GUI    
    def change_screen(self, screen_name):
        screen_manager = self.root.ids[
            'screen_manager']
        screen_manager.transition = CardTransition()
        screen_manager.transition.direction = 'up'
        screen_manager.transition.duration = .3
        screen_manager.current = screen_name    
MainApp().run()
