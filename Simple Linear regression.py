import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Salary_Data.csv')

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1:].values

from sklearn.model_selection import train_test_split
'''
test_size : 검증용 데이터 비율 
random_state: 데이터 분할시 셔플이 이루어 질 때를 위한 시드값 
'''
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
'''
* fit_intercept : 상수항 결정 (절편계산 X)
주로 데이터셋이 원점에 맞춰있을 때 사용
*normalize : 매개변수 무시 여부
'''
regressor=LinearRegression(fit_intercept=True, normalize=True, n_jobs=None)
#predict salary y_train
regressor.fit(x_train,y_train)

# Predicting the Test set results / really salary
y_prd=regressor.predict(x_test)

#TRAIN
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary & Experience (train)')
plt.xlabel('Experinece')
plt.ylabel('Salary')
plt.legend(['model','actual values'])
plt.show()

#TEST
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary & Experience (test)')
plt.xlabel('Experinece')
plt.ylabel('Salary')
plt.legend(['model','actual values'])
plt.show()

