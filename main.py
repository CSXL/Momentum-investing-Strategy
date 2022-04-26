
class QMS:
    def __init__(self,stocksfile,sk_key):
        import pandas as pd
        from iexfinance.stocks import Stock 
        import requests 
        import numpy as np
        import math
        self.stocks = pd.read_csv(stocksfile)
        self.Ticker = self.stocks['Ticker']
        self.key = sk_key 
        self.Ticker = list(self.Ticker)
        self.Stock_data = 0
        self.self.Stock_data_HQM = 0
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

#pricelist = list(get_price_data(self.Ticker))

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
#yearchange = list(get_yearchange_data(self.Ticker))

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
#available_symbols_list=list(get_available_symbol_data(self.Ticker))

#print(len(available_symbols_list),len(yearchange),len(pricelist))
    def load_data(self,available_symbols_list,pricelist,yearchange):
        columns = ['Symbol',"price",'return in one year','Number of shares to purchase']
        Stock_data = pd.DataFrame(columns=columns)
        data = zip(available_symbols_list,pricelist,yearchange)


        for x,y,z in data:
            Stock_data = Stock_data.append(
        pd.Series(
          [
          str(x).split()[1], float(y),float(z),'N/A'
          ],index=columns
            ),ignore_index=True
            )

    def get_portfolio_data(self):
        potfolio = int(input('Enter a value : '))
        potfolio_size = (potfolio/len(self.Stock_data.index))
        for i in range(0,len(self.Stock_data.index),1):
            self.Stock_data.loc[i,'Number of shares to purchase'] = potfolio_size/self.Stock_data.loc[i,'price']

        print(self.Stock_data)


    #stats
    def get_cols_length(self):
        HQM_columns = ['symbols','price','Number of shares to buy','year1ChangeReturn','year1ChangeReturn percentile',"month6ChangeReturn",'month6ChangeReturn percentile','month3ChangeReturn','month3ChangeReturn percentile','month1ChangeReturn','month1ChangeReturn percentile','HQM score']
        self.Stock_data_HQM = pd.DataFrame(columns=HQM_columns)
        self.Stock_data_HQM
        print(len(HQM_columns))

#info = zip(list(Stock_data['Symbol']),list(Stock_data['price']),list(Stock_data['Number of shares to purchase']),list(Stock_data['return in one year']))
    def load_returns(self):
        for index in range(0,len(list(self.Stock_data['Symbol']))-1):
            try:
                label = self.Stock_data['Symbol'][index]
                label = str(label)
                data = requests.get('https://cloud.iexapis.com/stable/stock/market/batch/?types=stats,quote&symbols='+str(label)+'&token='+str(key)).json()
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




        for row in self.Stock_data_HQM.index:
            for x in self.chunk_of_return_features:
                a = self.Stock_data_HQM[str(x)+'ChangeReturn']
                b = self.Stock_data_HQM.loc[row,str(x)+'ChangeReturn']
                self.Stock_data_HQM.loc[row,str(x)+'ChangeReturn percentile'] = stats.percentileofscore(a,b)
        

        print(self.Stock_data_HQM['HQM score'])
    def get_HQM_score(self):
        for row in self.self.Stock_data_HQM.index:
            avgpercentile = []
            for t in self.chunk_of_return_features:
                avgpercentile.append(self.self.Stock_data_HQM.loc[row,t+'ChangeReturn percentile'])
        self.self.Stock_data_HQM.loc[row,'HQM score'] = float(np.mean(avgpercentile))
        self.Stock_data_HQM.sort_values(by=['HQM score'],ascending=False,inplace=True)
        print(self.self.self.Stock_data_HQM)

#scores = self.self.Stock_data_HQM['HQM score']
#self.self.Stock_data_HQM = self.self.Stock_data_HQM.loc[self.self.Stock_data_HQM['HQM score'] > np.mean(scores)]

#self.self.Stock_data_HQM.sort_values(by=['HQM score'],ascending=False,inplace=True)

#self.self.Stock_data_HQM.index   
    def QMS_process(self):
      pricelist = list(self.get_price_data(self.Ticker))
      yearchange = list(self.get_yearchange_data(self.Ticker))
      available_symbols_list=list(self.get_available_symbol_data(Ticker))
      print(len(available_symbols_list),len(yearchange),len(pricelist))
      self.load_data(available_symbols_list,pricelist,yearchange)
      self.get_portfolio_data()
      self.get_cols_length()
      self.load_returns()
      self.get_percentile()
      self.get_HQM_score()









#-----------------------------------------------------------------------------------------------------#
import pandas as pd
from iexfinance.stocks import Stock 
import requests 
import numpy as np
import math
stocks = pd.read_csv('sp_500_stocks.csv')
Ticker = stocks['Ticker']
key = 'sk_dc49730633c14f4e9387de35a02a5592' 
Ticker = list(Ticker)

def get_price_data(stock_ticker):
  data = []
  for i in stock_ticker:
    try:
      stock_data = Stock(str(i),output_format='pandas',token=key).get_quote()
      data.append(stock_data['change']+stock_data['previousClose'])
      print(i)
    except:
      print('Unfound stock')
  
QVS_dataframe_cols = [
                      'symbols',
                      'price',
                      'Number of shares to buy',
                      'price-to-earning ratio',
                      'price-to-earning ratio percentile',
                      "price-to-book-value ratio",
                      'price-to-book-value ratio percentile',
                      'price-to-sales ratio',
                      'price-to-sales ratio percentile',
                      'ev-to-EBITDA ratio',
                      'ev-to-EBITDA ratio percentile',
                      'ev-to-gross profit ratio',
                      'ev-to-gross profit ratio percentile',
                      'Multiples score'
]
QVS_dataframe = pd.DataFrame(columns=QVS_dataframe_cols)
cloud_API = 'Tsk_aea2572be36943778ca7e902754deebc'

def chunk(lst,n):
  for i in range(0,len(lst),n):
    yield lst[i:i+n]
    
symbol_group = list(chunk(Ticker,502))
symbol_string = []
for i in range(0,len(symbol_group)):
  symbol_string.append(','.join(symbol_group[i]))
 # print(symbol_string[i])

symbol_string[0]

def book_and_sales_multiples():
  stock_list = []
  price_to_book_array =[]
  price_to_sales_array =[]
  price_to_earning_array = []
  for stock in 	symbol_string[0].split(","):
    try:
    #  val = si.get_stats_valuation(stock)
 
    #  val = val.iloc[:,:2]
 
#      val.columns = ["Attribute", "Recent"]
 #     price_to_earning = float(val[val.Attribute.str.contains("Trailing P/E")].iloc[0,1])
  #    price_to_book = float(val[val.Attribute.str.contains("Price/Book")].iloc[0,1])
   #   price_to_sales = float(val[val.Attribute.str.contains("Price/Sales")].iloc[0,1])
      data = requests.get('https://cloud.iexapis.com/stable/stock/market/batch/?types=advanced-stats,stats,quote&symbols='+str(stock)+'&token='+str(key)).json()
      price_to_sales = data[stock]['advanced-stats']['priceToSales']
      price_to_book = data[stock]['advanced-stats']['priceToBook']
      price_to_earning = data[stock]['quote']['peRatio']
      price_to_book_array.append(price_to_book)
      price_to_sales_array.append(price_to_sales)
      stock_list.append(stock)
      price_to_earning_array.append(price_to_earning)
      print(stock)
      print(price_to_book,price_to_sales,price_to_earning)
    except:
      print('Unfound stock')
      price_to_book_array.append(0)
      price_to_sales_array.append(0) 
      stock_list.append(stock)
      price_to_earning_array.append(0)
  return stock_list,price_to_book_array,price_to_sales_array,price_to_earning_array

stocks,book_ratio,sales_ratio,earning_ratio = book_and_sales_multiples()


print(len(stocks),len(earning_ratio),len(stocks),len(sales_ratio))
QVS_dataframe['symbols'] = stocks
QVS_dataframe['price-to-earning ratio'] = earning_ratio
QVS_dataframe['price-to-book-value ratio'] = book_ratio
QVS_dataframe['price-to-sales ratio']=sales_ratio

#data = requests.get('https://cloud.iexapis.com/stable/stock/market/batch/?types=advanced-stats,stats,quote&symbols='+str('AAPL')+'&token='+str("Tsk_dfa0c966233d465bb823aac1c2d58a22"))
#QVS_dataframe['ev-to-EBITDA ratio']
for values in range(0,len(QVS_dataframe['ev-to-gross profit ratio']),1):
  try:
    stocks = QVS_dataframe['symbols'][values]
    data = requests.get('https://cloud.iexapis.com/stable/stock/market/batch/?types=advanced-stats,stats,quote&symbols='+str(stocks)+'&token='+str(key)).json()
    QVS_dataframe.loc[values,'ev-to-gross profit ratio'] = (data[stocks]['advanced-stats']['enterpriseValue']/data[stocks]['advanced-stats']['grossProfit'])
    print(stocks)
  except:
    print('Unfound data')
    QVS_dataframe.loc[values,'ev-to-gross profit ratio'] = 0 

#data = requests.get('https://cloud.iexapis.com/stable/stock/market/batch/?types=advanced-stats,stats,quote&symbols='+str('AAPL')+'&token='+str("Tsk_dfa0c966233d465bb823aac1c2d58a22"))
#QVS_dataframe['ev-to-EBITDA ratio']
for values in QVS_dataframe.index:
  try:
    stock = QVS_dataframe['symbols'][values]
    #QVS_dataframe.loc[values,'ev-to-EBITDA ratio'] = ev_to_ebitda_array_1[values]
    data = requests.get('https://cloud.iexapis.com/stable/stock/market/batch/?types=advanced-stats,stats,quote&symbols='+str(stock)+'&token='+str(key)).json()
    QVS_dataframe.loc[values,'ev-to-EBITDA ratio'] = float(data[stock]['advanced-stats']['enterpriseValue']/data[stock]['advanced-stats']['EBITDA'])
  except:
    QVS_dataframe.loc[values,'ev-to-EBITDA ratio']  = 0
    print('Unfound data')

features = ['price-to-earning ratio','price-to-book-value ratio','price-to-sales ratio','ev-to-EBITDA ratio',"ev-to-gross profit ratio"]

for feature in features:
  QVS_dataframe[feature].fillna(0)  
QVS_dataframe


index = 0
for data in QVS_dataframe['ev-to-EBITDA ratio']:
  try:
    if data == 0:
      stock = QVS_dataframe.loc[index,'symbols']
      print(stock)
      batch_api = requests.get('https://cloud.iexapis.com/stable/stock/market/batch/?types=advanced-stats,stats,quote&symbols='+str(stock)+'&token='+str("Tsk_dfa0c966233d465bb823aac1c2d58a22")).json()
      QVS_dataframe.loc[index,'ev-to-EBITDA ratio'] = float(batch_api[stock]['advanced-stats']['enterpriseValue']/batch_api[stock]['advanced-stats']['EBITDA'])
    index +=1
  except:
    print(stock,'Unfound data')

def fill_holes(self,df,feature,api,jsonkey1,jsonkey2):
 while (df.loc[df[feature]==0]):
   for i in df.index:
     stock = df.loc[i,'symbols']
     if(df.loc[i,feature] == 0):
       try:
         batch_api = requests.get(api).json()
         df.loc[i,feature] = batch_api[jsonkey1][jsonkey2]
       except:
         print('could not do process')
 nan_df = (df.loc[df[feature]]==0)
 for i in nan_df.index:
   stock = nan_df.loc[i,'symbols']
   if(nan_df.loc[i,feature] == 0):
     try:
       batch_api = requests.get(feature).json()
       nan_df.loc[i,feature] = batch_api[stock][jsonkey1][jsonkey2]
     except:
       print('could not do proces')
 for i in list(nan_df.index):
   df.loc[i,feature] = nan_df.loc[i,feature]



while (QVS_dataframe.loc[QVS_dataframe['price-to-earning ratio']==0]):
  for i in QVS_dataframe.index:
    stock = QVS_dataframe.loc[i,'symbols']
    if(QVS_dataframe.loc[i,"price-to-earning ratio"] == 0):
      try:

        batch_api = requests.get('https://cloud.iexapis.com/stable/stock/market/batch/?types=advanced-stats,stats,quote&symbols='+str(stock)+'&token='+str(cloud_API)).json()
        QVS_dataframe.loc[i,"price-to-earning ratio"] = batch_api[stock]['quote']['peRatio']
      except:
        print('could not do process')

nan_df = (QVS_dataframe.loc[QVS_dataframe['price-to-earning ratio']==0])
for i in nan_df.index:
    stock = nan_df.loc[i,'symbols']
    if(nan_df.loc[i,"price-to-earning ratio"] == 0 ):
      try:

        batch_api = requests.get('https://cloud.iexapis.com/stable/stock/market/batch/?types=advanced-stats,stats,quote&symbols='+str(stock)+'&token='+str(cloud_API)).json()
        nan_df.loc[i,"price-to-earning ratio"] = batch_api[stock]['quote']['peRatio']
      except:
        print('could not do process')
nan_df
for i in list(nan_df.index):
  QVS_dataframe.loc[i,'price-to-earning ratio'] = nan_df.loc[i,'price-to-earning ratio']

while (QVS_dataframe.loc[QVS_dataframe['price-to-book-value ratio']==0]):
  for i in QVS_dataframe.index:
    stock = QVS_dataframe.loc[i,'symbols']
    if(QVS_dataframe.loc[i,"price-to-book-value ratio"] == 0):
      try:

        batch_api = requests.get('https://cloud.iexapis.com/stable/stock/market/batch/?types=advanced-stats,stats,quote&symbols='+str(stock)+'&token='+str(cloud_API)).json()
        QVS_dataframe.loc[i,"price-to-book-value ratio"] = batch_api[stock]['advanced-stats']['priceToBook']
      except:
        print('could not do process')
nan_df = (QVS_dataframe.loc[QVS_dataframe['price-to-book-value ratio']==0])
for i in nan_df.index:
    stock = nan_df.loc[i,'symbols']
    if(nan_df.loc[i,"price-to-book-value ratio"] == 0 ):
      try:

        batch_api = requests.get('https://cloud.iexapis.com/stable/stock/market/batch/?types=advanced-stats,stats,quote&symbols='+str(stock)+'&token='+str(cloud_API)).json()
        nan_df.loc[i,"price-to-book-value ratio"] = batch_api[stock]['advanced-stats']['priceToBook']
      except:
        print('could not do process')
nan_df
for i in list(nan_df.index):
  QVS_dataframe.loc[i,'price-to-book-value ratio'] = nan_df.loc[i,'price-to-book-value ratio']

while (QVS_dataframe.loc[QVS_dataframe['ev-to-gross profit ratio']==0]):
  for i in QVS_dataframe.index:
    stock = QVS_dataframe.loc[i,'symbols']
    if(QVS_dataframe.loc[i,"ev-to-gross profit ratio"] == 0):
      try:

        batch_api = requests.get('https://cloud.iexapis.com/stable/stock/market/batch/?types=advanced-stats,stats,quote&symbols='+str(stock)+'&token='+str(cloud_API)).json()
        QVS_dataframe.loc[i,"ev-to-gross profit ratio"] =( batch_api[stock]['advanced-stats']['enterpriseValue']/batch_api[stock]['advanced-stats']['grossProfit'])
      except:
        print('could not do process')
nan_df = (QVS_dataframe.loc[QVS_dataframe['ev-to-gross profit ratio']==0])
for i in nan_df.index:
    stock = nan_df.loc[i,'symbols']
    if(nan_df.loc[i,"ev-to-gross profit ratio"] == 0 ):
      try:

        batch_api = requests.get('https://cloud.iexapis.com/stable/stock/market/batch/?types=advanced-stats,stats,quote&symbols='+str(stock)+'&token='+str(cloud_API)).json()
        nan_df.loc[i,"ev-to-gross profit ratio"] = (batch_api[stock]['advanced-stats']['enterpriseValue']/batch_api[stock]['advanced-stats']['grossProfit'])
      except:
        print('could not do process')
nan_df
for i in list(nan_df.index):
  QVS_dataframe.loc[i,'ev-to-gross profit ratio'] = nan_df.loc[i,'ev-to-gross profit ratio']

while (QVS_dataframe.loc[QVS_dataframe['price-to-sales ratio']==0]):
  for i in QVS_dataframe.index:
    stock = QVS_dataframe.loc[i,'symbols']
    if(QVS_dataframe.loc[i,"price-to-sales ratio"] == 0):
      try:

        batch_api = requests.get('https://cloud.iexapis.com/stable/stock/market/batch/?types=advanced-stats,stats,quote&symbols='+str(stock)+'&token='+str(cloud_API)).json()
        QVS_dataframe.loc[i,"price-to-sales ratio"] = batch_api[stock]['advanced-stats']['priceToSales']
      except:
        print('could not do process')

nan_df = (QVS_dataframe.loc[QVS_dataframe['price-to-sales ratio']==0])
for i in nan_df.index:
    stock = nan_df.loc[i,'symbols']
    if(nan_df.loc[i,"price-to-sales ratio"] == 0 ):
      try:

        batch_api = requests.get('https://cloud.iexapis.com/stable/stock/market/batch/?types=advanced-stats,stats,quote&symbols='+str(stock)+'&token='+str(cloud_API)).json()
        nan_df.loc[i,"price-to-sales ratio"] = batch_api[stock]['advanced-stats']['priceToSales']
      except:
        print('could not do process')
nan_df
for i in list(nan_df.index):
  QVS_dataframe.loc[i,'price-to-sales ratio'] = nan_df.loc[i,'price-to-sales ratio']

QVS_dataframe.columns

#percentile calculation
features_1 = ["price-to-earning ratio","price-to-book-value ratio","price-to-sales ratio","ev-to-EBITDA ratio","ev-to-gross profit ratio"]
import scipy
for row in QVS_dataframe.index:
  for feature in features_1:
    a = QVS_dataframe[feature]
    b = QVS_dataframe.loc[row,feature]
    print(feature+str('percentile'))
    QVS_dataframe.loc[row,feature+str(' percentile')] = scipy.stats.percentileofscore(a,b)

QVS_dataframe['Multiples score']

for row in QVS_dataframe.index:
  avgpercentile = []
  for feature in features_1:
    avgpercentile.append(QVS_dataframe.loc[row,feature+str(' percentile')])
  QVS_dataframe.loc[row,'Multiples score'] = float(np.mean(avgpercentile))
  
QVS_dataframe

QVS_dataframe=QVS_dataframe.loc[QVS_dataframe['Multiples score']>np.mean(QVS_dataframe['Multiples score'])]
QVS_dataframe

price_info = list(get_price_data(Ticker))

pi = []
for price in price_info:
  __price__ = price
  __price__ = float(__price__)
  print(__price__)
  pi.append(__price__)

QVS_dataframe['price'] = pi

portfolio_budget = float(input("How much is your portfolio budget?"))
portfolio_size = float(portfolio_budget/len(QVS_dataframe.index))
portfolio_size
for i in QVS_dataframe.index:
  QVS_dataframe.loc[i,'Number of shares to buy'] = portfolio_size/QVS_dataframe.loc[i,'price']

QVS_dataframe = QVS_dataframe.sort_index(by=['Multiples score'],ascending=False,inplace=True)
QVS_dataframe