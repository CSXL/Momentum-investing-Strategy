import pandas as pd
from iexfinance.stocks import Stock 
import requests 
import numpy as np
import math
class QMS:
    def __init__(self,stocksfile,sk_key):

        self.stocks = pd.read_csv(stocksfile)
        self.Ticker = self.stocks['Ticker']
        self.key = sk_key 
        self.Ticker = list(self.Ticker)
        self.Stock_data = 0
        self.Stock_data_HQM = 0
    def get_price_data(self,stock_ticker):

        data = []
        for i in stock_ticker:
            try:
                stock_data = Stock(str(i),output_format='pandas',token=self.key).get_quote()
                data.append(stock_data['change']+stock_data['previousClose'])
                print(i)
            except:
                print('Unfound stock')
  
    
        return list(data)


    def get_yearchange_data(self,stock_ticker):
        data = []
        for i in stock_ticker:
            try:
                stock_data = Stock(str(i),output_format='pandas',token=self.key).get_quote()
                data.append(stock_data['ytdChange'])
                print(i)
            except:
                print('unfound stock')
    
        return list(data)

    def get_available_symbol_data(self,stock_ticker):
        data = []
        for i in stock_ticker:
            try:
                stock_data = Stock(str(i),output_format='pandas',token=self.key).get_quote()
                data.append(stock_data['symbol'])
                print(i)
            except:
                print('unfound stock')
    
        return list(data)

    def load_data(self,available_symbols_list,pricelist,yearchange):
        columns = ['Symbol',"price",'return in one year','Number of shares to purchase']
        self.Stock_data = pd.DataFrame(columns=columns)
        data = zip(available_symbols_list,pricelist,yearchange)


        for x,y,z in data:
            self.Stock_data = self.Stock_data.append(
        pd.Series(
          [
          str(x).split()[1], float(y),float(z),'N/A'
          ],index=columns
            ),ignore_index=True
            )

    def get_portfolio_data(self):
        potfolio = int(input('Enter a value : '))
        potfolio_size = (potfolio/70)
        for i in range(0,len(self.Stock_data.index),1):
            self.Stock_data.loc[i,'Number of shares to purchase'] = potfolio_size/self.Stock_data.loc[i,'price']

        print(self.Stock_data)


    def get_cols_length(self):
        HQM_columns = ['symbols','price','Number of shares to buy','year1ChangeReturn','year1ChangeReturn percentile',"month6ChangeReturn",'month6ChangeReturn percentile','month3ChangeReturn','month3ChangeReturn percentile','month1ChangeReturn','month1ChangeReturn percentile','HQM score']
        self.Stock_data_HQM = pd.DataFrame(columns=HQM_columns)
        self.Stock_data_HQM
        print(len(HQM_columns))

    def load_returns(self):
        for index in range(0,len(list(self.Stock_data['Symbol']))-1):
            try:
                label = self.Stock_data['Symbol'][index]
                label = str(label)
                data = requests.get('https://cloud.iexapis.com/stable/stock/market/batch/?types=stats,quote&symbols='+str(label)+'&token='+str(self.key)).json()
                self.Stock_data_HQM.loc[index,'symbols'] = self.Stock_data['Symbol'][index]
                self.Stock_data_HQM.loc[index,'price'] = self.Stock_data['price'][index]
                self.Stock_data_HQM.loc[index,'Number of shares to buy'] = self.Stock_data['Number of shares to purchase'][index]
                self.Stock_data_HQM.loc[index,'year1ChangeReturn'] = data[label]['stats']['ytdChangePercent']
                self.Stock_data_HQM.loc[index,'month6ChangeReturn'] = data[label]['stats']['month6ChangePercent']
                self.Stock_data_HQM.loc[index,'month3ChangeReturn'] = data[label]['stats']['month3ChangePercent']
                self.Stock_data_HQM.loc[index,'month1ChangeReturn'] = data[label]['stats']['month1ChangePercent']
                print(label)

            except:
                print('unfound stock')

#percentile calculation
#year1ChangeReturn month6ChangeReturn  month3ChangeReturn  month1ChangeReturn
    def get_percentile(self):
        from scipy import stats
        self.chunk_of_return_features = ['year1','month6','month3','month1']

       
        for time_period in self.chunk_of_return_features:

          if time_period == "year1":

            for row in self.Stock_data_HQM.index:
              if self.Stock_data_HQM.loc[row,str(time_period)+'ChangeReturn'] == None:
                label = self.Stock_data_HQM.loc[row,'symbols']
                data = requests.get('https://cloud.iexapis.com/stable/stock/market/batch/?types=stats,quote&symbols='+str(label)+'&token='+str(self.key)).json()
                self.Stock_data_HQM.loc[row,'year1ChangeReturn'] = data[label]['stats']['ytdChangePercent']

          if time_period == 'month6':

            for row in self.Stock_data_HQM.index:
              if self.Stock_data_HQM.loc[row,str(time_period)+'ChangeReturn'] == None:
                label = self.Stock_data_HQM.loc[row,'symbols']
                data = requests.get('https://cloud.iexapis.com/stable/stock/market/batch/?types=stats,quote&symbols='+str(label)+'&token='+str(self.key)).json()
                self.Stock_data_HQM.loc[row,'month6ChangeReturn'] = data[label]['stats']['month6ChangePercent']

          if time_period == 'month3':

            for row in self.Stock_data_HQM.index:
              if self.Stock_data_HQM.loc[row,str(time_period)+'ChangeReturn'] == None:
                label = self.Stock_data_HQM.loc[row,'symbols']
                data = requests.get('https://cloud.iexapis.com/stable/stock/market/batch/?types=stats,quote&symbols='+str(label)+'&token='+str(self.key)).json()
                self.Stock_data_HQM.loc[row,'month3ChangeReturn'] = data[label]['stats']['month3ChangePercent']

          if time_period == 'month1':

            for row in self.Stock_data_HQM.index:
              if self.Stock_data_HQM.loc[row,str(time_period)+'ChangeReturn'] == None:
                label = self.Stock_data_HQM.loc[row,'symbols']
                data = requests.get('https://cloud.iexapis.com/stable/stock/market/batch/?types=stats,quote&symbols='+str(label)+'&token='+str(self.key)).json()
                self.Stock_data_HQM.loc[row,'month1ChangeReturn'] = data[label]['stats']['month1ChangePercent']

        for row in self.Stock_data_HQM.index:
          for time_period in self.chunk_of_return_features:
            if self.Stock_data_HQM.loc[row,str(time_period)+'ChangeReturn'] == None:
              self.Stock_data_HQM.loc[row,str(time_period)+'ChangeReturn'] = 0.0

        for row in self.Stock_data_HQM.index:
            for x in self.chunk_of_return_features:
                a = self.Stock_data_HQM[str(x)+'ChangeReturn']
                b = self.Stock_data_HQM.loc[row,str(x)+'ChangeReturn']
                self.Stock_data_HQM.loc[row,str(x)+'ChangeReturn percentile'] = stats.percentileofscore(a,b)
        

        print(self.Stock_data_HQM['HQM score'])
    def get_HQM_score(self):
      for row in self.Stock_data_HQM.index:

        avgpercentile = []
        for t in self.chunk_of_return_features:
            avgpercentile.append(self.Stock_data_HQM.loc[row,t+'ChangeReturn percentile'])
        self.Stock_data_HQM.loc[row,'HQM score'] = float(np.mean(avgpercentile))
        self.Stock_data_HQM = self.Stock_data_HQM.sort_values(by=['HQM score'],ascending=False)
      print(self.Stock_data_HQM)


    def QMS_process(self):
      pricelist = list(self.get_price_data(self.Ticker))
      yearchange = list(self.get_yearchange_data(self.Ticker))
      available_symbols_list=list(self.get_available_symbol_data(self.Ticker))
      print(len(available_symbols_list),len(yearchange),len(pricelist))
      self.load_data(available_symbols_list,pricelist,yearchange)
      self.get_portfolio_data()
      self.get_cols_length()
      self.load_returns()
      self.get_percentile()
      self.get_HQM_score()
__key__ = str(input('Enter IEX key'))        
at = QMS('sp_500_stocks.csv',__key__)
at.QMS_process()
at.Stock_data_HQM[:70]
