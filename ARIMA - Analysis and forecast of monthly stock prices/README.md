# 1. Introduction

While studying, I got really into data analysis - especially time series. This project is one I worked on using GRETL, where I used ARIME and SARIMA models to forecast sales.It was a great way to turn theory into something more hands-on and actually see how those models work in practice.

Feel free to check it out and see what insights I found!

# 2. Abstract

In this project my main goal is to use time series of interest to analyze and forecast. I have chosen monthly stock prices of a Polish video game developer, publisher and distributor CD Projekt S.A. Firstly, I will introduce data and tell a little bit about important dates which were affecting this time series. Later I will conclude if this time series can be considered as a representation of stationary process. Next I will analyze correlogram and also use automatic criteria in order to identify test models. Then I will estimate those models using the training set and also check them, that is establish if residuals of those models can be considered as a realization of white noise process and also I will check whether the residuals are normally distributed or not. Finally I will forecast using those models and I will compare them to the real values of the test set. At the end I will summarize my observations

## 3. Introducing the data

Let’s first have a look at our data. This series is containing monthly stock prices of CD Projekt S.A. from January 2015 to December 2020. The currency of those prices is in Polish currency PLN. PLN to EUR converter is 0.22 on the day 20 January 2021.

![Power BI - Sales Dashboard](images/image1.png)
