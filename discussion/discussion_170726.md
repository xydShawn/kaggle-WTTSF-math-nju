## 任务目标
针对一部分维基网页，利用20150701-20161231每天的访问量数据，预测20170101-20170301这段时间每天网页的访问量

## 注意事项
1. 结果需要将所有网页60天的预测结果降成一个单一序列(series)提交，序列的index需要结合key_1.csv得到

## 观察发现
1. Page按照后缀分有四大类  
org\_all-access\_spider  
org\_desktop\_all-agents  
org\_mobile-web\_all-agents  
org\_all-access\_all-agents 
2. Page中有一部分表示的是语种
3. 同一个搜索内容可能有多种Page与之对应

## 模型算法
1. 时间序列
2. 线性回归
3. RNN(GNU, LSTM)
