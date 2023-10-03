## Loading data and packages

#Load packages
library(gridExtra)
library(smooth)
library(fpp2)
library(zoo)
library(dygraphs)
library(readxl)
#Reading the data
coffeeprice <- read_excel("coffee (1).xlsx", 
                          skip = 10)
#change the columns names
names(coffeeprice)<-c("date","price","ppi_coffee","ppi_tea","bean_price","temperature","er_real")

#define the data as a ts object
tsdata = ts(coffeeprice,start=1980,frequency = 12)

## Code - Using the quarterly data 

# Quarterly data
price<-tsclean(aggregate(tsdata[,"price"], nfrequency = 4, mean))
ppi_coffee<-tsclean(aggregate(tsdata[,"ppi_coffee"], nfrequency = 4, mean))
ppi_tea<-tsclean(aggregate(tsdata[,"ppi_tea"], nfrequency = 4, mean))
bean_price<-tsclean(aggregate(tsdata[,"bean_price"], nfrequency = 4, mean))
temperature<-tsclean(aggregate(tsdata[,"temperature"], nfrequency = 4, mean))
er_real<-tsclean(aggregate(tsdata[,"er_real"], nfrequency = 4, mean))
autoplot(decompose(price))
d1q = ggAcf(price, lag.max = 100)
d2q = ggPacf(price, lag.max = 500)

## Training the model without the last 2 years:

train_price <- window(price, end = 2020-1/4)
test_price <- window(price, start = 2020)
flength<-length(test_price)
XREG <- cbind(ppi_coffee,ppi_tea,bean_price,temperature,er_real)
train_XREG = window(XREG, end = 2020-1/4)
test_XREG  = window(XREG, start = 2020)

#training & forecasting naive
ts.naive = naive(train_price, h=flength)
#training & forecasting average
ts.ave   = meanf(train_price, h=flength) 
#training & forecasting drift
ts.drift = rwf(train_price, drift=TRUE, h=flength)
#training & forecasting simple moving avg
ts.sma   = sma(train_price, h=flength)
#training & forecasting simple exponential smoothing
ts.ses   = ses(train_price, h=flength)
#training & forecasting ets: error+trend+seasonality
fit.ets = forecast::ets(train_price)
ts.ets = forecast::forecast.ets(fit.ets, h=flength)
#training & forecasting arma
fit.arma = auto.arima(train_price, d=0, seasonal=FALSE)
ts.arma  = forecast::forecast(fit.arma, h=flength)
#training & forecasting arima
fit.arima = auto.arima(train_price, seasonal=FALSE)
ts.arima  = forecast::forecast(fit.arima, h=flength)
#training & forecasting sarima
fit.sarima = auto.arima(train_price)
ts.sarima = forecast::forecast(fit.sarima, h=flength) 
#training dynamic linear reg
dlr.fit = auto.arima(train_price, xreg = train_XREG)
#forecasting dynamic linear reg
dlr.fc = forecast::forecast(dlr.fit, xreg = test_XREG )
fit2 = auto.arima(train_price, xreg = fourier(train_price, K=2), seasonal=FALSE)
fc2  = forecast(fit2, xreg = fourier(train_price, K=2, h = flength))
fit_nn = nnetar(train_price)
price.nn= forecast(fit_nn, h = flength)


p2q<-autoplot(window(price, start = 2018)) +
  ggtitle("Data Quarterly-2Y")+
  autolayer(dlr.fc$mean, series="DLR")+
  autolayer(ts.drift$mean, series="Drift") +
  autolayer(ts.ses$mean, series="SES, SMA, Naive")+
  autolayer(ts.ets$mean, series="ETS")+
  autolayer(ts.arma$mean, series="ARMA")+
  autolayer(ts.arima$mean, series="ARIMA, SARIMA") +
  autolayer(ts.sarima$mean, series="SARIMA")+
  autolayer(ts.ave$mean, series="Average")+
  autolayer(fc2$mean, series="DHR")+
  autolayer(price.nn$mean, series = "NN")

mape.naiveq = accuracy(ts.naive$mean, test_price)[5]
mape.aveq = accuracy(ts.ave$mean, test_price)[5]
mape.driftq = accuracy(ts.drift$mean, test_price)[5]
mape.smaq = accuracy(ts.sma$forecast, test_price)[5]
mape.sesq = accuracy(ts.ses$mean, test_price)[5]
mape.etsq = accuracy(ts.ets$mean, test_price)[5]
mape.armaq = accuracy(ts.arma$mean, test_price)[5]
mape.arimaq = accuracy(ts.arima$mean, test_price)[5]
mape.sarimaq = accuracy(ts.sarima$mean, test_price)[5]
mape.dlrq = accuracy(dlr.fc$mean, test_price)[5]
mape.dhrq = accuracy(fc2$mean, test_price)[5]
mape.nnq = accuracy(price.nn$mean, test_price)[5]

pfit1q = auto.arima(train_price, xreg = fourier(train_price, K=1), seasonal=FALSE)
pfc1q  = forecast(pfit1q, xreg = fourier(train_price, K=1, h = flength))
pfit2q = auto.arima(train_price, xreg = fourier(train_price, K=2), seasonal=FALSE)
pfc2q  = forecast(pfit2q, xreg = fourier(train_price, K=2, h = flength))

# aicc DHR - 2 years
pfit1q$aicc
pfit2q$aicc

## Training the model without the last 1 years:

train_price <- window(price, end = 2021-1/4)
test_price <- window(price, start = 2021)
flength<-length(test_price)
XREG <- cbind(ppi_coffee,ppi_tea,bean_price,temperature,er_real)
train_XREG = window(XREG, end = 2021-1/4)
test_XREG  = window(XREG, start = 2021)

#training & forecasting naive
ts.naive = naive(train_price, h=flength)
#training & forecasting average
ts.ave   = meanf(train_price, h=flength) 
#training & forecasting drift
ts.drift = rwf(train_price, drift=TRUE, h=flength)
#training & forecasting simple moving avg
ts.sma   = sma(train_price, h=flength)
#training & forecasting simple exponential smoothing
ts.ses   = ses(train_price, h=flength)
#training & forecasting ets: error+trend+seasonality
fit.ets = forecast::ets(train_price)
ts.ets = forecast::forecast.ets(fit.ets, h=flength)
#training & forecasting arma
fit.arma = auto.arima(train_price, d=0, seasonal=FALSE)
ts.arma  = forecast::forecast(fit.arma, h=flength)
#training & forecasting arima
fit.arima = auto.arima(train_price, seasonal=FALSE)
ts.arima  = forecast::forecast(fit.arima, h=flength)
#training & forecasting sarima
fit.sarima = auto.arima(train_price)
ts.sarima = forecast::forecast(fit.sarima, h=flength) 
#training dynamic linear reg
dlr.fit = auto.arima(train_price, xreg = train_XREG)
#forecasting dynamic linear reg
dlr.fc = forecast::forecast(dlr.fit, xreg = test_XREG )
fit21 = auto.arima(train_price, xreg = fourier(train_price, K=2), seasonal=FALSE)
fc2  = forecast(fit21, xreg = fourier(train_price, K=2, h = flength))
fit_nn = nnetar(train_price)
price.nn= forecast(fit_nn, h = flength)


p1q<-autoplot(window(price, start = 2018)) +
  ggtitle("Data Quarterly-1Y")+
  autolayer(dlr.fc$mean, series="DLR")+
  autolayer(ts.drift$mean, series="Drift") +
  autolayer(ts.ses$mean, series="SES, SMA, Naive")+
  autolayer(ts.ets$mean, series="ETS")+
  autolayer(ts.arma$mean, series="ARMA")+
  autolayer(ts.arima$mean, series="ARIMA, SARIMA") +
  autolayer(ts.sarima$mean, series="SARIMA")+
  autolayer(ts.ave$mean, series="Average")+
  autolayer(fc2$mean, series="DHR")+
  autolayer(price.nn$mean, series = "NN")

mape.naive1q = accuracy(ts.naive$mean, test_price)[5]
mape.ave1q = accuracy(ts.ave$mean, test_price)[5]
mape.drift1q = accuracy(ts.drift$mean, test_price)[5]
mape.sma1q = accuracy(ts.sma$forecast, test_price)[5]
mape.ses1q = accuracy(ts.ses$mean, test_price)[5]
mape.ets1q = accuracy(ts.ets$mean, test_price)[5]
mape.arma1q = accuracy(ts.arma$mean, test_price)[5]
mape.arima1q = accuracy(ts.arima$mean, test_price)[5]
mape.sarima1q = accuracy(ts.sarima$mean, test_price)[5]
mape.dlr1q = accuracy(dlr.fc$mean, test_price)[5]
mape.dhr1q = accuracy(fc2$mean, test_price)[5]
mape.nn1q = accuracy(price.nn$mean, test_price)[5]

pfit11q = auto.arima(train_price, xreg = fourier(train_price, K=1), seasonal=FALSE)
pfc11q  = forecast(pfit11q, xreg = fourier(train_price, K=1, h = flength))
pfit21q = auto.arima(train_price, xreg = fourier(train_price, K=2), seasonal=FALSE)
pfc21q  = forecast(pfit21q, xreg = fourier(train_price, K=2, h = flength))

# aicc DHR - 1 year
pfit11q$aicc
pfit21q$aicc

# Quarterly data
price<-tsclean(aggregate(tsdata[,"price"], nfrequency = 4, mean))
ppi_coffee<-tsclean(aggregate(tsdata[,"ppi_coffee"], nfrequency = 4, mean))
ppi_tea<-tsclean(aggregate(tsdata[,"ppi_tea"], nfrequency = 4, mean))
bean_price<-tsclean(aggregate(tsdata[,"bean_price"], nfrequency = 4, mean))
temperature<-tsclean(aggregate(tsdata[,"temperature"], nfrequency = 4, mean))
er_real<-tsclean(aggregate(tsdata[,"er_real"], nfrequency = 4, mean))
autoplot(decompose(price))

fit2 = auto.arima(price, xreg = fourier(price, K=2), seasonal=FALSE)
fc2  = forecast(fit2, xreg = fourier(price, K=2, h = 8))

# find outliers and replace

price2<-na.interp(aggregate(tsdata[,"price"], nfrequency = 4, mean))
tsoutliers(price2)
#outliers
fit2_out = auto.arima(price2, xreg = fourier(price2, K=2), seasonal=FALSE)
fc2_out  = forecast(fit2_out, xreg = fourier(price2, K=2, h = 8))

autoplot(window(price2, start = 2015)) +
  xlab("Quarters") +
  ylab("Grounded Coffee Price")+
  autolayer(fc2_out$mean, series="With Outliers")+
  autolayer(fc2$mean, series="Without Outliers")+
  ylim(3.5,6.5)


autoplot(window(price, start = 2018)) +
  ggtitle("Data Quarterly-2Y")+
  autolayer(fc2)

autoplot(window(price, start = 2015)) +
  xlab("Quarters") +
  ylab("Grounded Coffee Price")+
  autolayer(fc2)

checkresiduals(fc2)

fit2
fc2