import datreant.core as dtr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import re
import matplotlib.dates as mdatesi
import yaml
from statsmodels.tsa.arima_model import ARIMA
from pandas.tools.plotting import autocorrelation_plot
from sklearn.preprocessing import StandardScaler
from math import factorial
from sklearn.metrics import mean_squared_error

class PatientData:
    def __init__(self,filePath):
        self.BuildPatientData(filePath)
        self.order = None
        self.err_least = float("inf")
        self.best_order = None
        self.dirPath = filePath.rstrip('data-%s'%re.findall(r'/d+',filePath))

    def BuildPatientData(self,filePath):
        if '.csv' in filePath:
            return None
        if 'data-' not in filePath:
            return None

        dateTime = []
        with open(filePath,'r') as f:
            for line in f:
                index1 = line.index('\t')
                date = line[:index1]
                index2 = line[index1+1:].index('\t') + index1
                time = line[index1+1:index2+1]
                dateTime.append('%s %s'%(date,time))
         
        DateTime = [pd.datetime.strptime(dateT,"%m-%d-%Y %H:%M") for dateT in dateTime]
        df = pd.read_csv(filePath, sep='\t', names =['Date', 'Time', 'Code', 'Measurement'])
        
        date = df['Date']
        df = df.drop('Date',axis=1)
        df = df.drop('Time',axis=1)
        
        self.patientDFDateTime = df.set_index(pd.DataFrame(DateTime,columns=['DateTime'])['DateTime'])
        
        self.patientDFDateTime['Measurement'] = self.patientDFDateTime['Measurement'].astype(float)

        self.patientDFDateTimeRoll = self.patientDFDateTime

        self.patientDFDateTimeRoll['Measurement'] = PatientData.savitzky_golay(self.patientDFDateTimeRoll['Measurement'].rolling(window=20).mean(),51,3)
       
    @staticmethod
    def savitzky_golay(y, window_size, order, deriv=0, rate=1):
        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order+1)
        half_window = (window_size -1) // 2
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
    
        return np.convolve( m[::-1], y, mode='valid')

    def DataSplit(self):
        codeVals = [33,34,35,48,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72]

        codeValDataDict = {}
        codeValDataDictRoll = {}
        for codeVal in codeVals:
            codeValData = self.patientDFDateTime[self.patientDFDateTime.Code == codeVal]
            codeValDataRoll = self.patientDFDateTimeRoll[self.patientDFDateTimeRoll.Code == codeVal]
            if np.shape(codeValData)[0] <= 0:
                continue
            codeValDataDict[codeVal] = codeValData
            codeValDataDictRoll[codeVal] = codeValDataRoll

        self.codeValDataDict = codeValDataDict
        self.codeValDataDictRoll = codeValDataDictRoll

    def TimeSeriesPlots(self):
        for key,value in self.codeValDataDict.items():
            fig = plt.figure()
            ax = fig.add_subplot(111)
            value.drop('Code',axis=1).plot(ax=ax)
            self.codeValDataDictRoll[key].drop('Code',axis=1).plot(ax=ax)
            fig.savefig('%sCode-%s-TimeSeries.pdf'%(self.dirPath,key))
            del(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.patientDFDateTime.drop('Code',axis=1).plot(ax=ax)
        self.patientDFDateTimeRoll.drop('Code',axis=1).plot(ax=ax)
        fig.savefig('%s-All-TimeSeries.pdf'%self.dirPath)
        del(fig)

    def autocorrelationPlots(self):
        for key,value in self.codeValDataDict.items():
            fig = plt.figure()
            ax = fig.add_subplot(111)
            pd.tools.plotting.autocorrelation_plot(value.drop('Code',axis=1),ax=ax)
            pd.tools.plotting.autocorrelation_plot(self.codeValDataDictRoll[key].drop('Code',axis=1),ax=ax)
            fig.savefig('%sCode-%s-Auto.pdf'%(self.dirPath,key))
            del(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        pd.tools.plotting.autocorrelation_plot(self.patientDFDateTime.drop('Code',axis=1),ax=ax)
        pd.tools.plotting.autocorrelation_plot(self.patientDFDateTimeRoll.drop('Code',axis=1,),ax=ax)
        fig.savefig('%s-All-Auto.pdf'%self.dirPath)
        del(fig)

    def ApplyArima(self):
        for key in self.codeValDataDictRoll.keys():
            self.ArimaPredict(key)
    
        self.ArimaPredict()

    def ArimaPredict(self,key=None):
        predictions = list()
        if key is not None:
            X = self.codeValDataDictRoll[key].drop('Code',axis=1).dropna(axis=0).values
        else:
            X = self.patientDFDateTimeRoll.drop('Code',axis=1).dropna(axis=0).values
        
        if len(X) <= 25:
            return

        size = int(len(X)*.66)
        train, test = X[0:size],X[size:len(X)]
        history = [x for x in train]
        for t in range(len(test)):
            model=ARIMA(history,order=self.order)
            model_fit = model.fit(disp=0)
            residuals = pd.DataFrame(model_fit.resid)
            describe_resid = residuals.describe()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
        error = mean_squared_error(test,predictions)
        fig = plt.figure()
        ax = fig.add_subplot(111)

        if key is not None:
            trainPre = self.codeValDataDictRoll[key].drop('Code',axis=1).dropna(axis=0)
        else:
            trainPre = self.patientDFDateTimeRoll.drop('Code',axis=1).dropna(axis=0)
        
        trainPre.plot(ax=ax,color='red')
        ax.plot(predictions,color='blue')
        ax.plot(test,color='green')
        if key is not None:
            fig.savefig('%sArimaPredictCode-%s-Order-%s.pdf'%(self.dirPath,key,self.order))
        else:
            fig.savefig('%sArimaPredict-All-Order-%s.pdf'%(self.dirPath,self.order))
        del(fig)

        if error < self.err_least:
            self.err_least = error
            self.best_order = self.order

    def ArimaForeCast(self,bestOrder):
        predictions = list()
        train = self.patientDFDateTimeRoll.drop('Code',axis=1).dropna(axis=0).values
    
        if len(train) <= 25: 
            return

        try:
            size = len(train)
            history = [x for x in train]
            for t in range(int(len(train)*0.1)):
                model=ARIMA(history,order=bestOrder)
                model_fit = model.fit(disp=0)
                residuals = pd.DataFrame(model_fit.resid)
                describe_resid = residuals.describe()
                output = model_fit.forecast()
                yhat = output[0]
                predictions.append(yhat)

            fig = plt.figure()
            ax = fig.add_subplot(111)

            trainPre = np.array(self.patientDFDateTimeRoll.drop('Code',axis=1).dropna(axis=0))

            ax.plot(predictions,color='red')
            ax.plot(trainPre,color='blue')
            fig.savefig('%sArimaPredict-All-BestOrder-%s.pdf'%(self.dirPath,self.order))
            del(fig)
        except:
            pass


        














