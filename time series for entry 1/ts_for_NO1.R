setwd("E:\\files\\Kaggle\\Web Traffic Time Series Forecasting\\data")
rawdata<-read.csv("train_1.csv",header=T,fileEncoding = 'utf-8')
data1<-rawdata[1,]
plotdata<-data1[2:length(data1)]
pdata<-unlist(c(plotdata))
day<-c(1:length(pdata))
plot(day,pdata,type="l")
for(i in 1:2) print(Box.test(pdata,lag=6*i))
acf(pdata,lag=100)
pacf(pdata,lag=100)
library(tseries)
library(forecast)
tsdata<-ts(pdata,start=c(2015,182),frequency = 365)
#ARIMA模型
fit1<-auto.arima(tsdata)
summary(fit1)
for(i in 1:2) print(Box.test(fit1$residuals,lag=6*i))
for1<-forecast(fit1,h=60)
plot(for1)
L1<-for1$fitted-1.96*sqrt(fit1$sigma2)
U1<-for1$fitted+1.96*sqrt(fit1$sigma2)
L2<-ts(for1$lower[,2],start=c(2015,8),frequency=12)
U2<-ts(for1$upper[,2],start=c(2015,8),frequency=12)
c1<-min(dataplot,L1,L2)
c2<-max(dataplot,L2,U2)
plot(dataplot,type="p",pch=8,xlim=c(2000,2016))
lines(for1$fitted,col=2,lwd=2)
lines(for1$mean,col=2,lwd=2)
lines(L1,col=4,lty=2)
lines(U1,col=4,lty=2)
lines(L2,col=4,lty=2)
lines(U2,col=4,lty=2)
#条件异方差模型
library(FinTS)
for(i in 1:5) print(ArchTest(fit1$residuals,lag=i))
for(i in 1:5) print(Box.test(fit1$residuals^2,type="Ljung-Box",lag=i))
