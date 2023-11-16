import pandas as pd
import sys
import os
import numpy as np
import itertools
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from datetime import date, timedelta, time, datetime
from sklearn.linear_model import LinearRegression
from dataArchive import DataArchive
from datetime import date, timedelta, time, datetime

class TemperatureModel():
    """
    A class that models temperature based on historical temperature and barometric pressure.
    ...
    
    Attributes
    ----------
    data : pd DataFrame
        the temperature and barometric pressure data to be used in modeling
    start_index : int
        time of day in index form that the model will start
    start_temp: float
        temperature value the model will start at
    prediction_date : datetime date
        date in the dataframe that the model is built for
    predictions : arr
        the predicted temperatures for the prediction date starting at the start_index
    adjustor: float
        how much the model will adjust to the input temperature
    training_data: pd DataFrame
        the temperature data for model construction
        
    Methods
    -------
    plot_historical (model_date, plots):
        plots the previous *plots* days of temperature starting at a specified date
    predict():
        models temperature through the end of the prediction day
    set_adjustor(val):
        sets the adjustor
    set_sunset_time(sunset_time):
        sets the time of sunset/time to be predicted
    set_start_time(start_time):
        sets the model starting time of day
    get_adjustor():
        returns the adjustor  
    get_predictions():
        returns the predictions array 
    get_start_time():
        returns the model starting time of day 
    get_sunset_time():
        returns the time of sunset/time to be predicted
    """
    
    def __init__ (self, start_time = None, sunset_time = 18.5, adjustor = -99999, days = 4):
        """
        Class constructor
        
        Args:
            data (pd DataFrame): the temperature and barometric pressure data to be used for predictions
            csv (str): a csv to the data to be used for predictions if data is not provided
            start_time (float): the time of day that the model should start predicting from
            sunset_time (float): the time of day of sunset or that the model should predict
            adjustor (float): the amount the model should adjust to the current day's temperature data
            days (int): the number of days to be predicted by the model
        """
        data = self._get_data_archive()
        self._prep_data(data)
        if (type (start_time) == type (None)):
            self.start_index = int((self.data['ts'].dt.time.iloc[-1].hour * 1 + (self.data['ts'].dt.time.iloc[-1].minute / 60))*4.6)
        else:
            self.start_index = int(start_time *4.6)
            
        if(adjustor == -99999):
            self.adjustor = (0.3471 + (0.0353 * (self.start_index / 4.6)))
        else:
            self.adjustor = (adjustor)
        
        try:
            self.sunset_time = int (sunset_time * 6)
        except:
            print ("The provided sunset time could not be converted to an index")
            print ("Use set_sunset_time() to change sunset time")
            self.sunset_time = 185
            
        self.prediction_date = self.data['ts'].dt.date.iloc[-1]
        
        self._get_time()
        
        try:
            self.start_temp = self.data[(self.data['ts'].dt.date == self.prediction_date)].iloc[self.start_index]['OutTemp']
        except:
            self.start_temp = self.data[(self.data['ts'].dt.date == self.prediction_date)].iloc[-1]['OutTemp'] 
            
        self.prediction_days = days
    
    def _get_data_archive(self):
        tel = #tel num
        dataArch = #data archive from tel num
        daynum = #number of days of data
        channel_t = #temperature channel name
        channel_p = #pressure channel name
        name, xt, yt = dataArch.getData(channel_t, (datetime.now() - timedelta(days = daynum)), nrSecs=86400 * daynum)
        name, xp, yp = dataArch.getData(channel_p, (datetime.now() - timedelta(days = daynum)), nrSecs=86400 * daynum)

        dft = np.array([xt,yt])
        dfp = np.array([xp,yp])
        dft = self.resample_temp(dft)
        dfp = self.resample_temp(dfp)
        dfc = pd.DataFrame(list(dft), columns = ('ts','Time','OutTemp'))
        dfc1 = pd.DataFrame(list(dfp), columns = ('ts','Time','OutPressure'))
        shiftamt = len(dfc) - len(dfc1)
        dfc['OutPressure'] = dfc1['OutPressure']
        dfc['OutPressure'] = dfc['OutPressure'].shift(shiftamt)
        dfc['ts'] = pd.to_datetime(dfc['ts']) - timedelta(hours = 10)
        dfc['Time'] = (dfc['ts'].dt.hour * 60. + dfc['ts'].dt.minute) / 60.
        return dfc
            
    def _update_start_temp(self, model_date):
        if(model_date == date(1,1,1)):
            model_date = self.prediction_date
        return self.data[(self.data['ts'].dt.date == model_date)].iloc[self.start_index]['OutTemp']
        
        
    def get_adjustor (self):
        """
        Returns the adjustor
        The adjustor is the percent the model will adjust to the updated temperature data provided by the current day

        Args:
            none

        Returns:
            float: The adjustor
        """
        
        return self.adjustor
    
    def set_adjustor (self, val):
        """
        Sets the adjustor
        The adjustor is the percent the model will adjust to the updated temperature data provided by the current day
        To get rid of the adjustor set the adjustor to 1
        
        Args:
            val (int): the updated value for the adjustor
            
        Returns:
            none
        """
        
        self.adjustor = val
    
    def get_sunset_time(self):
        """
        Returns the time of sunset or the time to be predicted for
        
        Args:
            none

        Returns:
            time: The time of sunset 
        """
        
        return time(hour = int (self.sunset_time // 10), minute = int (abs ((self.sunset_time // 10) - self.sunset_time / 10) * 60))
    
    def set_sunset_time(self, sunset_time):
        """
        Sets the time of sunset/time to be predicted
        
        Args:
            sunset_time (float): the updated value for the sunset time
            
        Returns:
            none
        """
        
        try:
            self.sunset_time = int (sunset_time * 10)
        except:
            print ("The provided sunset time could not be converted to an index")
            print ("Use set_sunset_time() to change sunset time")
            self.sunset_time = 185
            
    def get_start_time(self):
        """
        Returns the model start time
        
        Args:
            none

        Returns:
            time: The start time
        """
        
        return time(hour = int (self.start_time // 10), minute = int (abs ((self.start_time // 10) - self.start_time / 10) * 60))
         
    def set_start_time(self, start_time):
        """
        Sets the time of day that the model will start predicting from
        
        Args:
            start_time (float): the updated value for the start time
            
        Returns:
            none
        """
        
        try:
            self.start_index = int (start_time * 10)
        except:
            print ("The provided model start time could not be converted to an index")
        
    def _get_training_data (self, model_date = date(1, 1, 1)):
        if(model_date == date(1, 1, 1)):
            model_date = self.prediction_date
        self.training_data = self.data[(self.data.ts.dt.date <= model_date - timedelta(days = 1)) & (self.data.ts.dt.date >= model_date - timedelta(days = 3))]
        if (len (self.training_data) < 720):
            self.training_data = self.data[(self.data['ts'].dt.date <= model_date)].iloc[-720: ]
            
    
    def _get_time (self, day_length = 112):
        self.times = []
        self.times = np.arange(0,24,(24/day_length))
    
    def resample_temp(self, data, bin_size_secs=600):
        """
        Resamples the data into bins of bin_size_secs.
        Returns new list.
        """

        rows = []
        t0 = data[0][0]
        tref = t0
        cnt = 0
        sum0 = 0
        secs = 0
        for row in range(len(data[0])):
            ts, temp = data[0][row],data[1][row]
            dt = ts - t0
            if dt >= bin_size_secs:
                secs = t0 + bin_size_secs // 2
                hour = (secs - tref) / 3600
                new_row = datetime.utcfromtimestamp(secs).strftime('%Y-%m-%d %H:%M:%S'), hour, sum0 / cnt
                rows.append(new_row)
                sum0 = temp
                cnt = 1
                t0 = ts
            else:
                cnt += 1
                sum0 += temp
        if cnt > 1:
            secs = t0 + bin_size_secs // 2
            hour = (secs - tref) / 3600
            rows.append((datetime.utcfromtimestamp(secs).strftime('%Y-%m-%d %H:%M:%S'), hour, sum0 / cnt))
        return rows

            
    def _prep_data (self, data):
        self.data = None
        try:
            data.replace('', np.nan, inplace = True)
            data.dropna(inplace = True) 
            data = data.filter(items = ("ts", "OutTemp", "OutPressure", "Time"))
        except:
            print ("The data that was provided does not have the correct columns")
        self.data = pd.DataFrame.copy(data)
    
     
        
    def _my_trapez (self, x, amp, mu, top, bottom, slope, bg):
        """
        Uses Trapezoidal integration to model x based on given parameters

        Args:
            x (arr): The time values to be predicted by the function
            amp (float): The amplitude
            mu (float): The average of our distribution
            top (float): The width of the amplitude
            bottom (float): The width of the bottom of the function
            slope (float): The slope of the linear part of the function
            bg (float): The intercept of the linear part of the function

        Returns:
            arr: The predicted temperatures over the range of x
        """
        
        def fx (v):
            v = np.abs(v)
            y = np.where (v > bottom, 0, np.where (v > top, m * v + b, amp))
            return y
        m = -amp * 1.0 / (bottom - top)
        b = - m * top + amp
        nfx = np.vectorize (fx)
        return nfx (x-mu) + slope * x + bg
    
    def _my_trapez_start (self, x, amp, mu, top, bottom, slope, bg, start = 0):
        """
        Uses Trapezoidal integration to model x based on given parameters
        
        Args:
            x (arr): The time values to be predicted by the function
            amp (float): The amplitude
            mu (float): The average of our distribution
            top (float): The width of the amplitude
            bottom (float): The width of the bottom of the function
            slope (float): The slope of the linear part of the function
            bg (float): The intercept of the linear part of the function
            start (float): The starting temperature of the function

        Returns:
            arr: The predicted temperatures over the range of x adjusted to the model start
        """
        
        if start == 0:
            return self._my_trapez(x, amp, mu, top, bottom, slope, bg)
        else:
            def fx (v):
                v = np.abs(v)
                y = np.where (v > bottom, 0, np.where (v > top, m * v + b, amp))
                return y
            m = -amp * 1.0 / (bottom - top)
            b = - m * top + amp
            nfx = np.vectorize (fx)
            return self._smooth((start + (nfx (x - mu) + slope * x + bg) - (nfx (0 - mu) + slope * 0 + bg)),16)
        
    def _combined_bp (self, x, amp, mu, top, bottom, slope, bg, start, bpw):
        """
        Composite model for temperature based on barometric pressure

        Args:
            x (arr): The time values to be predicted by the function
            amp (float): The amplitude
            mu (float): The average of our distribution
            top (float): The width of the amplitude
            bottom (float): The width of the bottom of the function
            slope (float): The slope of the linear part of the function
            bg (float): The intercept of the linear part of the function
            start (float): The starting temperature of the function

        Returns:
            arr: The predicted temperature over the range of x adjusted for barometric pressure
        """
        
        return self._my_trapez_start (x, amp, mu, top, bottom, slope, bg, start)+ self._bp_function (x, bpw)
    
    def _smooth (self, arr, wsize):
        """
        Smooths an array over a given range

        Args:
            arr (array): The array to be smoothed
            wsize (int): The range of the array to be smoothed

        Returns:
            arr: The smooth array
        """
        
        iarr  = np.cumsum (arr)
        half = wsize // 2
        a = iarr[0: -wsize]
        b = iarr[wsize:]
        m = (b - a) / wsize
        if (len (arr) % 2 == 0):
            out = np.copy(arr)
        else:
            out = np.copy(arr)[1: ]
            return self._smooth(out, wsize)
        out[half: -half] = m
        return out
    
    def _bp_function (self, xTest, bpw):
        """
        Linear function for barometric pressure

        Args:
            xTest (array): the time range to be modeled over
            bpw (float): the adjustment to be made with respect to time

        Returns:
            arr: The barometric pressure adjustment to predicted temperature over the length of xTest
        """
        
        bp = np.zeros(len (xTest))
        bp[0] = 0
        i = 1
        while i < len (bp):
            bp[i] = bp[i - 1] + (bpw / 24)
            i += 1
        if (len (bp) % 2 != 0):
            bp = bp[1:]
        return bp
    
    def _get_start (self, model_date = date(1, 1, 1)):
        if(model_date == date(1, 1, 1)):
            model_date = self.prediction_date
        start = float(self.data[(self.data.ts.dt.date == (model_date))]['OutTemp'].iloc[0])
        return start
    
    def _get_slope (self, model_date = date(1, 1, 1)):
        if(model_date == date(1, 1, 1)):
            model_date = self.prediction_date
        self._get_training_data(model_date)
        bp_modeling = self.training_data[self.training_data.ts.dt.date == (model_date - timedelta(days = 1))]
        x_bp = bp_modeling[['Time']]
        y_bp = bp_modeling['OutPressure']
        lm = LinearRegression()
        lm.fit(np.array(x_bp), np.array(y_bp))
        return (lm.coef_[0]) / 2
    
    def predict (self, model_date = date(1, 1, 1),dl = -1, st = 0):
        """
        Models temperature through the end of the prediction day

        Args:
            none

        Returns:
            arr: array of predictions
        """
        # start_temp = self.data[(self.data['ts'].dt.date == model_date)].iloc[self.start_index]['OutTemp']
        # print(f"Starting Temp Before: {start_temp}")
        start_temp = self._smooth(np.array(self.data[(self.data['ts'].dt.date == model_date)]['OutTemp']),14)[self.start_index-1]
        # print(f"Starting Temp After: {start_temp}")
        if(dl == -1):
            dl = len(self.data[(self.data.ts.dt.date == model_date)])
        else:
            dl = dl
        if(model_date == date(1, 1, 1)):
            model_date = self.prediction_date
        self.predictions = []
        day_new = []
        self._get_time(day_length = dl)
        self._get_training_data (model_date)
        start_temp = self._update_start_temp (model_date)
        x = self._smooth (np.array (self.training_data['Time']), 8)
        y = self._smooth (np.array (self.training_data['OutTemp']), 8)
        start = self._get_start (model_date)
        slope = self._get_slope (model_date)
        params2 = np.max(y) - np.mean(y), 12.1875, 1.4, 4.4314, 0.07644953, 3.748
        sciParams2, sciCov2 = curve_fit (self._my_trapez_start, np.ravel(x), np.ravel(y), params2)
        day = self._combined_bp (np.ravel(self.times), *sciParams2, start, slope)
        day_new = np.copy (day)
        adj = (start_temp - day_new[self.start_index]) * self.adjustor
        for i in range(len(day)):
            day_new[i] = day[i] + adj
        times = np.array(self.data[self.data.ts.dt.date == model_date]['ts'])
        df = pd.DataFrame(day_new[:-2])
        if(len(times) < 100):
            times = np.array(self.data[self.data.ts.dt.date == (model_date - timedelta(days = 1))]['ts'].iloc[self.start_index:] + timedelta(days = 1))
            df = pd.DataFrame(day_new[self.start_index:-2])
        df['ts'] = times[0:len(df)]
        return df.iloc[st:]
        
                  
        
    def _daterange(self, start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)
        
    def plot_historical (self, model_date = date(1, 1, 1), plots = -1, filename = "TemperaturePrediction.html"):
        """
        Plots the previous *plots* days of temperature starting at a specified date
        Args:
            model_date (date): the end date of predictions
            plots (int): the number of days to be plotted
            
        Returns:
            none
        """
        if (plots == -1):
            plots = self.prediction_days
        else:
            plots = plots
        model_date = self.prediction_date
        predictions,actuals,times, dates = [], [], [], []
        df_cdp = self.predict(model_date = model_date, dl = 112)
        self.start_index = 46
        for single_date in self._daterange ((model_date - timedelta(days = plots)), (model_date)):
            df_cdp = pd.concat((df_cdp, self.predict(model_date = single_date, st = 46)))
            dates.append(single_date)
        actuals = self.data['OutTemp'].iloc[-(112*plots+self.start_index):]
        times = self.data['ts'].iloc[-(112*plots+self.start_index):]
        actuals = pd.DataFrame(actuals)
        actuals['ts'] = times
        df_model = actuals
        df_cdp.columns = ["Prediction","ts"]
        fig1 = px.scatter(df_model, x = 'ts', y = 'OutTemp', color_discrete_sequence = ['#63C3E1'], symbol_sequence = ['circle-open'], trendline = "lowess", trendline_options = dict(frac = (0.05)))
        fig2 = px.scatter(df_cdp,x = 'ts', y = 'Prediction',color_discrete_sequence=['#D09184'],symbol_sequence= ['triangle-up-open'])
        fig = go.Figure(data = fig2.data + fig1.data)
        fig.update_layout(
            showlegend = True,
            width = (700 + (100 * plots)),
            height = 500,
            title="Outside Temperature Prediction",
            xaxis_title="Date",
            yaxis_title="Temperature",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#4a148c"
                )
            )
        pio.write_html(
        fig, file=filename, auto_open=False, full_html=False, include_plotlyjs=True
        )
        fig.show()

if __name__ == "__main__":
    import argparse
    def parseArguments(in_args):
        description = "Predicts outside temperature"
        usage = "\n{} [-o outName] [-p hours] [-m modelDays] [-s startTime] [-a adjustor]\n".format(in_args[0])
        epilog = "file located at: \"TemperaturePrediction.html\""

        parser = argparse.ArgumentParser(description=description, usage=usage, epilog=epilog)
        parser.add_argument(
            "-o", "--outname", dest="outname", help="Output HTML file for graph", type=str, default="TemperaturePrediction.html"
        )
        parser.add_argument("-p", "--hours", dest="predTime", help="Time of day in float from 0-24 to predict", type=float, default=18.5)
        parser.add_argument("-m", "--modelDays", dest="nDays", help="Number of days to include", type=int, default=5)
        parser.add_argument("-a", "--adjustor", dest="adj", help="How much to adjust the model at the start (from 0-1)", type=float, default=.7)
        args = None
        try:
            args = parser.parse_args(in_args[1:])
        except Exception as e:
            print(e)
            parser.print_help()
            sys.exit(0)
        return args
    args = parseArguments(sys.argv)
    tm = TemperatureModel(sunset_time = args.predTime, days = args.nDays, adjustor = args.adj)
    tm.plot_historical(filename = args.outname)