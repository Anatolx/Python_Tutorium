# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 09:52:41 2021

@author: Lorenz
"""

# Aufgabe 1: Physikprotokoll Pendelschwingung

import numpy as np
from matplotlib import pyplot as plt

periodendauer=[10.11,10.23,10.34,10.28,10.26,10.24,10.28,10.46,10.27,10.38]

def test_max(x):
    if x==np.max(periodendauer):
        print("Das ist korrekt!")
    elif x=='code':
        print('np.max(periodendauer)')
    else :
        print('Das stimmt so nicht, bitte nochmal probieren!')
        
def test_min(x):
    if x==np.min(periodendauer):
        print("Das ist korrekt!")
    elif x=='code':
        print('np.min(periodendauer)')
    else :
        print('Das stimmt so nicht, bitte nochmal probieren!')
        
def test_mean(x):
    if x==np.mean(periodendauer):
        print("Das ist korrekt!")
    elif x=='code':
        print('np.mean(periodendauer)')
    else :
        print('Das stimmt so nicht, bitte nochmal probieren!')
        
def test_std(x):
    if x==np.std(periodendauer):
        print("Das ist korrekt!")
    elif x=='code':
        print('np.std(periodendauer)')
    else :
        print('Das stimmt so nicht, bitte nochmal probieren!')
        
def test_plot(x):
    if x == 'plot':
        fig1 = plt.figure()
        plt.boxplot(periodendauer)
        plt.scatter([1,1,1,1,1,1,1,1,1,1], periodendauer)
    elif x == 'code':
        print('fig1 = plt.figure()\n  plt.boxplot(periodendauer)\n  plt.scatter([1,1,1,1,1,1,1,1,1,1], periodendauer)')
    
def test_einlesen():
    print("data = pd.read_csv('Messergebnisse.csv') \n data")
def test_plot2(data):
    fig2 = plt.figure()
    plt.plot(data['t[s]'],data['x[m]'])
    print("fig2 = plt.figure() \n plt.plot(data['t[s]'],data['x[m]'])")
    
def test_plot3(data):
    fig3 = plt.figure()
    plt.plot(data['t[s]'],data['x[m]']*1000)
    print("fig3 = plt.figure() \n plt.plot(data['t[s]'],data['x[m]']*1000)")
    
def test_plot4(data):
    fig4 = plt.figure()
    plt.plot(data['t[s]'],data['x[m]']*1000)
    plt.xlabel('$t$ in s')
    plt.ylabel('$x$ in mm')
    print("fig4 = plt.figure() \n plt.plot(data['t[s]'],data['x[m]']*1000) \n plt.xlabel('$t$ in s') \n plt.ylabel('$x$ in mm')")
    
def test_plot5(data):
    fig5 = plt.figure()
    plt.plot(data['t[s]'],data['x[m]']*1000)
    plt.xlabel('$t$ in s')
    plt.ylabel('$x$ in mm')
    plt.xlim([0,10])
    print("fig5 = plt.figure() \n plt.plot(data['t[s]'],data['x[m]']*1000)\n plt.xlabel('$t$ in s')\n plt.ylabel('$x$ in mm')\n plt.xlim([0,10])")

def test_fit(data):
    from scipy import optimize

    def test_func(x, a, b):
        return a * np.cos(b * x)
    
    params, params_covariance = optimize.curve_fit(test_func, data['t[s]'], data['x[m]']*1000,
                                                   p0=[1, 10])
    
    print(params)
    print(" from scipy import optimize\n def test_func(x, a, b): \n      return a * np.cos(b * x) \n    params, params_covariance = optimize.curve_fit(test_func, data['t[s]'], data['x[m]']*1000,\n  p0=[1, 10])\n print(params)")
    
    
def test_plot6(data, test_func, params):
    fig6 = plt.figure()
    plt.plot(data['t[s]'],data['x[m]']*1000, label='Messwerte')
    plt.xlabel('$t$ in s')
    plt.ylabel('$x$ in mm')
    plt.xlim([0,10])
    plt.plot(data['t[s]'], test_func(data['t[s]'], params[0], params[1]),
             label='Fitted function')
    plt.legend()
    print("""fig6 = plt.figure()
    plt.plot(data['t[s]'],data['x[m]']*1000, label='Messwerte')
    plt.xlabel('$t$ in s')
    plt.ylabel('$x$ in mm')
    plt.xlim([0,10])
    plt.plot(data['t[s]'], test_func(data['t[s]'], params[0], params[1]),
             label='Fitted function')
    plt.legend())""")
    
    
# Aufgabe 2
def test2_plot1(masse, auslenkung):
    plt.figure()
    plt.scatter(masse,auslenkung)
    plt.xlabel('$m$ in g')
    plt.ylabel('$\Delta x$ in mm')
    print("""plt.figure()
          plt.scatter(masse,auslenkung)
          plt.xlabel('$m$ in g')
          plt.ylabel('$\Delta x$ in mm')""")
  
def test2_fit(masse, auslenkung):
    x=masse
    y=auslenkung
    linear_model=np.polyfit(x,y,1)
    linear_model_fn=np.poly1d(linear_model)
    x_s=np.arange(0,300)
    
    plt.figure()
    plt.scatter(masse,auslenkung)
    plt.xlabel('$m$ in g')
    plt.ylabel('$\Delta x$ in mm')
    plt.plot(x_s,linear_model_fn(x_s),color="green")
    
    print("""x=masse
y=auslenkung
linear_model=np.polyfit(x,y,1)
linear_model_fn=np.poly1d(linear_model)
x_s=np.arange(0,300))
    
    
plt.figure()
plt.scatter(masse,auslenkung)
plt.xlabel('$m$ in g')
plt.ylabel('$\Delta x$ in mm')
plt.plot(x_s,linear_model_fn(x_s),color="green")

linear_model""")

def test2_ber(linear_model):
    k=1/linear_model[0]*9.81
    k
    print("""k=1/linear_model[0]*9.81
k""")