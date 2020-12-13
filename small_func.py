#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 17:47:03 2020

@author: francescoserraino
"""
import webbrowser
    

class test_func():
    def save(self, x):
        with open("surname.txt", "w") as fobj:
            fobj.write(str(x))
    #import necessary library
    def web_linker(self,link):
        webbrowser.open(link)
            
            
print(open('surname.txt'))

with open("surname.txt") as fobj:
    print(fobj)
       
    
with open('surname.txt') as f:
    lines = f.read()
    

