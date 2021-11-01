#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import stats
import numpy as np


# In[2]:


ipl_df=pd.read_csv('IPL IMB381IPL2013.csv')


# In[3]:


ipl_df


# In[4]:


type(ipl_df)


# In[5]:


ipl_df.head()


# In[6]:


ipl_df.columns


# In[7]:


list(ipl_df.columns)


# In[8]:


ipl_df.transpose()


# In[9]:


ipl_df.shape


# In[10]:


ipl_df.info()


# In[11]:


ipl_df[2:10]


# In[12]:


ipl_df[-5:]


# In[13]:


ipl_df['premium'] = ipl_df['SOLD PRICE']-ipl_df['BASE PRICE']


# In[14]:


df=ipl_df[['premium']].sort_values('premium',ascending=True)


# In[15]:


df.head()


# In[ ]:





# In[16]:


ipl_df[['PLAYER NAME','BASE PRICE','SOLD PRICE','premium']].sort_values('premium',ascending=False)


# In[17]:


ipl_df.groupby('AGE')['SOLD PRICE'].mean().reset_index()


# In[18]:


df=ipl_df.groupby(['AGE','PLAYER NAME'])['AGE'].count()


# In[19]:


df.tail()


# In[20]:


ipl_df[ipl_df['SIXERS']>80][['PLAYER NAME','SIXERS']]


# In[ ]:





# # AUTOS_DF

# In[1]:


autos_df = pd.read_csv('auto-mpg.data')


# In[206]:


autos_df.head()


# In[207]:


autos_df.columns = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','origin','name']


# In[208]:


autos_df.head()


# In[209]:


autos_df.info()


# In[ ]:


autos_df


# In[ ]:


autos_df[0:400]


# In[ ]:


autos_df[29:35]


# In[ ]:


autos_df['horsepower'] = pd.to_numeric(autos_df['horsepower'],errors='coerce')


# In[ ]:


autos_df.info


# In[ ]:


autos_df[autos_df.horsepower.isnull()]


# In[ ]:


autos_df = autos_df.dropna(subset=['horsepower'])


# In[ ]:





# # BARCHART

# In[145]:


sn.barplot(x = 'AGE',y = 'SOLD PRICE', data = ipl_df)


# In[146]:


sn.barplot(x = 'AGE',y = 'SOLD PRICE', data = ipl_df, hue='PLAYING ROLE')


# # Histogram

# In[147]:


plt.hist(ipl_df['SOLD PRICE'],bins=15)


# In[148]:


sn.distplot(ipl_df['SOLD PRICE'])


# In[149]:


ipl_df[ipl_df['SOLD PRICE']>1350000.0][['PLAYER NAME','PLAYING ROLE','SOLD PRICE']]


# In[150]:


sn.distplot(ipl_df['SOLD PRICE'])


# In[151]:


sn.distplot(ipl_df[ipl_df['CAPTAINCY EXP']==1]['SOLD PRICE'],color ='y',label='CAPTAINCY EXP')
sn.distplot(ipl_df[ipl_df['CAPTAINCY EXP']==0]['SOLD PRICE'],color ='r', label='NO CAPTAINCY')
plt.legend()


# # Box plot/ Whisker plot

# lower quartile(1st)
# median
# upper quartile(3rd)
# IQR

# In[152]:


sn.boxplot(ipl_df['SOLD PRICE'])


# In[153]:


box=plt.boxplot(ipl_df['SOLD PRICE'])


# In[154]:


[item.get_ydata()[0] for item in box['caps']]


# In[155]:


[item.get_ydata()[0] for item in box['whiskers']]


# In[156]:


[item.get_ydata()[0] for item in box['medians']]


# In[157]:


sn.boxplot(x ='PLAYING ROLE',y='SOLD PRICE', data=ipl_df)


# # Scatter plot

# In[158]:


ipl_df_batsman = ipl_df[ipl_df['PLAYING ROLE']=='Batsman']


# In[159]:


plt.scatter(x = ipl_df_batsman.SIXERS ,y = ipl_df_batsman['SOLD PRICE'])


# # Regression plot

# In[160]:


sn.regplot(x = 'SIXERS', y = 'SOLD PRICE', data=ipl_df_batsman)


# # Pair plot

# In[161]:


influential_features = ['SR-B','AVE','SIXERS','SOLD PRICE']


# In[162]:


sn.pairplot(ipl_df[influential_features])


# In[163]:


ipl_df[influential_features].corr()


# # Heat map

# In[164]:


sn.heatmap(ipl_df[influential_features].corr(),annot = True)


# # PROBABILITY THEORY

# # It is observed that 10% of their customers return the items for many reasons. On a specific day, 20 customers purchased items from the shop. Calculate:

# # 1.The probability that exactly 5 customers will return the items.

# In[165]:


stats.binom.pmf(5,20,0.1)


# In[166]:


pmf_df=pd.DataFrame({'success':range(0,21),'pmf':list(stats.binom.pmf(range(0,21),20,0.1))})


# In[167]:


pmf_df.head(20)


# In[168]:


sn.barplot(x=pmf_df.success , y= pmf_df.pmf)
plt.ylabel('pmf')
plt.xlabel('Numbers of items recurred')


# # 2.The probability that a maximum of 5 customers will return the items.

# In[169]:


a=stats.binom.cdf(5,20,0.1)


# In[170]:


a


# In[171]:


cdf_df=pd.DataFrame({'success':range(0,6),'cdf':list(stats.binom.cdf(range(0,6),20,0.1))})


# In[172]:


cdf_df.head(20)


# In[173]:


sn.barplot(x=cdf_df.success , y= cdf_df.cdf)
plt.ylabel('cdf')
plt.xlabel('Numbers of items recurred')


# # 3.The probability that more than 5 customers will return the items.

# In[174]:


b=stats.binom.cdf(5,20,0.1)


# In[175]:


b


# In[176]:


1 - stats.binom.cdf(5, 20, 0.1)


# In[177]:


cdf_df2=pd.DataFrame({'success':range(5,21),'cdf':list(1-stats.binom.cdf(range(5,21),20,0.1))})


# In[178]:


cdf_df2.head()


# In[179]:


sn.barplot(x=cdf_df2.success , y= cdf_df2.cdf)
plt.ylabel('cdf2')
plt.xlabel('Numbers of items recurred')


# # 4.An average number of customers who are likely to return the items and the variance and the standard deviation of the number of returns.

# In[180]:


mean,var = stats.binom.stats(20, 0.1)
print('Average :', mean, 'Variance :', var)


# # Poisson distribution

# # Q.The number of cash withdrawals at an ATM follows a Poisson Distribution at 10 withdrawals per hour. Calculate:

# # 1.The probability that a maximum of 5 withdrawals will happen.

# In[181]:


stats.poisson.cdf(5, 10)


# # 2.The probability that a number of withdrawals over a period of 3 hours will exceed 30.

# In[182]:


# will be 30
#stats.poisson.cdf(30, 30)
#exceed 30
1-stats.poisson.cdf(30, 30)


# In[183]:


pmf_data_poisson = pd.DataFrame({'success' : range(0, 31), 'pmf' : list(stats.poisson.pmf(range(0,31), 10))})


# In[184]:


sn.barplot(x = pmf_data_poisson.success, y = pmf_data_poisson.pmf)
plt.xlabel('number of withdrawals')
plt.ylabel('pmf')


# In[185]:


stats.poisson.cdf(30, 25)


# In[186]:


1-stats.poisson.cdf(2,25,0.05)


# # Normal Distribution

# In[187]:


beml_df = pd.read_csv('BEML.csv')


# In[188]:


beml_df.head()


# In[189]:


beml_1_df = beml_df[['Date','Close']]


# In[190]:


beml_1_df.head()


# In[191]:


beml_2_df = beml_1_df.set_index(pd.DatetimeIndex(beml_1_df['Date']))


# In[192]:


beml_2_df.head()


# In[193]:


plt.figure(figsize=(10,6))
plt.scatter(beml_2_df.Date,beml_2_df.Close)
plt.xlabel('Time')
plt.ylabel('Close Price')


# In[194]:


beml_2_df['gain']= beml_2_df.Close.pct_change(periods=1)


# In[195]:


beml_2_df.head()


# In[196]:


beml_2_df= beml_2_df.dropna()


# In[197]:


beml_2_df.head()


# In[198]:


plt.plot(beml_2_df.index,beml_2_df.gain)
plt.xlabel('Time')
plt.ylabel('gain')


# In[199]:


glaxo_df=pd.read_csv('glaxo.csv')


# In[200]:


glaxo_df.head()


# In[201]:


glaxo_1_df=glaxo_df[['Date','Close']]


# In[202]:


glaxo_1_df.head()


# In[203]:


glaxo_1_df=glaxo_1_df.sort_index()


# In[54]:


glaxo_1_df.head()


# In[55]:


glaxo_1_df = glaxo_1_df.set_index(pd.DatetimeIndex(glaxo_1_df['Date']))


# In[56]:


glaxo_1_df.head()


# In[57]:


plt.figure(figsize=(10,6))
plt.scatter(glaxo_df.Date,glaxo_df.Close)
plt.xlabel('Time')

plt.ylabel('Close Price')


# In[58]:


# gain=[cp(t)-cp(t-1)]/cp(t-1)   .... done by pct
glaxo_1_df['gain']= glaxo_1_df.Close.pct_change(periods=1)


# In[59]:


glaxo_1_df.head()


# In[60]:


glaxo_1_df= glaxo_1_df.dropna()


# In[61]:


glaxo_1_df.head()


# In[62]:


plt.plot(glaxo_1_df.index,glaxo_1_df.gain)
plt.xlabel('Time')
plt.ylabel('gain')


# In[63]:


#distribution plot between two stock gain
sn.distplot(glaxo_1_df.gain,label='glaxo')
sn.distplot(beml_2_df.gain, label='beml')
plt.legend()


# In[64]:


print("Mean=",round(glaxo_1_df.gain.mean(),4))
print("std=",round(glaxo_1_df.gain.std(),4))


# In[65]:


print("Mean=",round(beml_2_df.gain.mean(),4))
print("std=",round(beml_2_df.gain.std(),4))
# beml is higher risk as std is high percentage


# In[66]:


beml_2_df.gain.describe()


# In[67]:


glaxo_1_df.gain.describe()


# # Confidence Interval

# In[68]:


glaxo_df_ci = stats.norm.interval(0.95,loc=glaxo_1_df.gain.mean(),scale=glaxo_1_df.gain.std())


# In[69]:


print("gain at 95% confidence interval=",np.round(glaxo_df_ci,4))


# In[70]:


beml_df_ci = stats.norm.interval(0.95,loc=beml_2_df.gain.mean(),scale=beml_2_df.gain.std())


# In[71]:


print("gain at 95% confidence interval=",np.round(beml_df_ci,4))


# # 4.Which stock has higher probability of making a daily return of 2% or more?

# In[72]:


beml_3_df = beml_2_df[['gain']]


# In[73]:


beml_3_df.head()


# In[ ]:





# # 5.Which stock has higher probability of making a loss (risk) of 2% or more?

# In[ ]:





# # Hypothesis Test

# In[75]:


passport_df=pd.read_excel('passport.xlsx')


# In[76]:


passport_df.head()


# In[77]:


print(list(passport_df.processing_time))


# In[78]:


import math
def z_test(pop_mean, pop_std, sample):
    z_score = (sample.mean() - pop_mean) / (pop_std / math.sqrt(len(sample)))
    return z_score  , stats.norm.cdf(z_score)                              


# In[79]:


z_test(30, 12.5, passport_df.processing_time)


# # Classwork on Z test 
# A bottle filling machine fills water into 5 liters (5000 cm3) bottles. The company wants to test the null hypothesis that the average amount of water filled by the machine into the bottle is at least 5000 cm3. A random sample of 60 bottles coming out of the machine was selected and the exact contents of the selected bottles are recorded. The sample mean was 4,998.1 cm3. The population standard deviation is known from the experience to be 1.30 cm3. Assume that the population is normally distributed with the standard deviation of 1.30 cm3. Write code to test the hypothesis at a of 5%. Explain the results.

# In[80]:


z=(4998.1 - 5000 ) / (1.30 / math.sqrt(60))


# In[81]:


z


# In[82]:


import math
def z_test(pop_mean, pop_std,sample,sample_mean):
    z_score = (sample_mean - pop_mean) / (pop_std / math.sqrt(sample))
    return z_score                               


# In[83]:


z_test(5000,1.30,60,4998.1)


# # Classwork on One Sample ttest
# A fabric manufacturer would like to understand the proportion of defective fabrics they produce. His shop floor staff have been stating that the percentage of defective is not more than 18%. He would like to test whether the claim made by his shop floor staff is correct. He picked up a random sample of 100 fabrics and found 22 defectives. Use a = 0.05 and write code to test the hypothesis that the percentage of defective components is less than 18%.
The t-test is used when the population standard deviation S is unknown (and hence estimated from the sample) and is estimated from the sample. Mathematically,
t-statistics = (X'-u)/S(n^1/2)
# In[ ]:





# # Classwork on Two Sample TTest
# Suppose that the makers of ABC batteries want to demonstrate that their battery lasts an average of at least 60 min longer than their competitor brand. Two independent random samples of 100 batteries of each kind are selected from both the brands, and the batteries are used continuously. The sample average life of ABC is found to be 450 min. The average life of competitor batteries is 368 min with the sample standard deviation of 82 min and 78 min, respectively. Frame a hypothesis and write the code to test ABC’s claim at 95% significance.

# In[ ]:





# In[ ]:





# # Linear Regression

# In[84]:


mba_df= pd.read_csv('MBA Salary.csv')


# In[85]:


mba_df.head()


# In[86]:


mba_df.info()


# # Creating a Features set(x) and output variable(y)

# In[87]:


import statsmodels.api as sm


# In[88]:


X = sm.add_constant(mba_df['Percentage in Grade 10'])
X.head()


# In[89]:


Y = mba_df['Salary']
Y.head()


# # Splitting the dataset into training and validation sets

# In[90]:


from sklearn.model_selection import train_test_split


# In[91]:


train_X,test_X,train_Y,test_Y =train_test_split(X, Y, train_size=0.8, random_state=100)


# # Fitting the Model 

# In[92]:


mba_info =sm.OLS(train_Y, train_X).fit()


# In[93]:


mba_info.params


# In[94]:


# Y=B(0)+B(1)X+E............simple linear regression model
# MBA Salary = 30587.285 + 3560.587*(percentage in grade 10)+E


# # Model Diagonostic

# 1.Coefficient of Determination (R-sqaured)
# 2.Hypothessis test for regression coefficient
# 3.Analysis of varianc for overall model validity (imp)
# 4 Residual Analysis to validates the regression model aasumption
# 5.Outliers Analysis

# In[95]:


#Coefficient of Determination (R-sqaured)
#SST(sum of sqaure of total variation)= SSR(sum of sqaureof unexplained error) + SSE(sum of sqaure of explained error)
# R -sqaured(coefficient of determination) = SSR/SST=1-(SSE/SST)  ......value ->(0-1)


# In[96]:


mba_info.summary2()


# # Hypothesis
# Ho:B1 Ha:B1!=0

# # Residual Analysis

# In[97]:


Actual value, -> Y
Predicted Value -> Y^
Y-Y^
1. The residuals are Normally distributed.
2.Variance of residual is constant (Homoscedascity).
3.The funcional form of regression is correctly specified.
4.There are no outliers.


# # Check for Normal Distribution

# # Probability Probability Plot (P-P Plot)

# In[98]:


mba_salary_resid=mba_info.resid


# In[99]:


probplot = sm.ProbPlot(mba_salary_resid)


# In[100]:


plt.figure(figsize=(10,8))
probplot.ppplot(line = '45')
plt.show()


# In[101]:


def get_standardized_values(vals):
    return (vals - vals.mean())/vals.std()


# In[102]:


plt.scatter(get_standardized_values(mba_info.fittedvalues),get_standardized_values(mba_salary_resid))
plt.title("Residual Plot : MBA Salary Prediction")
plt.xlabel("standardized Predicted Values")
plt.ylabel("standardized Residuals")


# # Outliers Analysis
1.Z-Score
2.Mahalanobis
3.Cook's Distance
4.Leverage Values

# In[103]:


#z-score
#z = (Y(i)^-Y)/variance of Y

from scipy.stats import zscore
mba_df['z_score_salary'] = zscore(mba_df.Salary)
mba_df[(mba_df.z_score_salary > 3.0)| (mba_df.z_score_salary<-3.0)]


# In[106]:


#cook's distance
mba_influence = mba_info.get_influence()
(c,p) = mba_influence.cooks_distance


# In[107]:


plt.stem(np.arange(len(train_X)), np.round(c,3), markerfmt =",")
plt.title("Cook's Distance")
plt.xlabel("Row Index")
plt.ylabel("Cook's Distance")


# In[108]:


#leverage Values
from statsmodels.graphics.regressionplots import influence_plot
fig, ax = plt.subplots( figsize=(8,6) ) 
influence_plot( mba_info , ax = ax )
plt.title("Figure 4.4 - Leverage Value Vs Residuals") 
plt.show()

#the size of the circle is proportional to the product of residual and leverage value. The larger the circle, the larger is the residual and hence influence of the observation.
# In[109]:


pred_y = mba_info.predict(test_X)

#Measuring Accuracy
1. Mean sqaure error(MSE)
2. Root mean square error(RMSE)
3. Mean absolute percentage error(MAPE)
# In[110]:


from sklearn.metrics import r2_score, mean_squared_error
np.abs(r2_score(test_Y,pred_y))


# # Classwork Linear Regression
# The dataset country.csv contains Corruption Perception Index and Gini Index of 20 countries. Corruption Perception Index close to 100 indicates low corruption and close to 0 indicates high corruption. Gini Index is a measure of income distribution among citizens of a country (high Gini indicates high inequality). Corruption Index is taken from Transparency International, while Gini Index is sourced from Wikipedia.
# 
# 1. Develop a simple linear regression model (Y = b0 + b1X) between corruption perception index (Y) and Gini index (X). What is the change in the corruption perception index for every one unit increase in Gini index?
# 
# 2. What proportion of the variation in corruption perception index is explained by Gini index?
# 
# 3. Is there a statistically significant relationship between corruption perception index and Gini index at alpha value 0.1?
# 
# 4. Calculate the 95% confidence interval for the regression coefficient b1.
# 

# In[111]:


country_df= pd.read_csv('country.csv')


# In[112]:


country_df.head()


# In[113]:


country_df.info()


# In[114]:


X = sm.add_constant(country_df['Corruption_Index'])
X.head()


# In[115]:


Y = country_df['Gini_Index']
Y.head()


# In[116]:


train_X,test_X,train_Y,test_Y =train_test_split(X, Y, train_size=0.8, random_state=100)


# In[117]:


country_info =sm.OLS(train_Y, train_X).fit()


# In[118]:


country_info.params


# In[119]:


country_info.summary2()


# In[120]:


country_Gini_Index_resid=country_info.resid


# In[121]:


probplot = sm.ProbPlot(country_Gini_Index_resid)


# In[122]:


plt.figure(figsize=(10,8))
probplot.ppplot(line = '45')
plt.show()


# In[123]:


plt.scatter(get_standardized_values(country_info.fittedvalues),get_standardized_values(country_Gini_Index_resid))
plt.title("Residual Plot : Gini_Index prediction")
plt.xlabel("standardized Predicted Values")
plt.ylabel("standardized Residuals")


# In[124]:


from scipy.stats import zscore
country_df['z_score_Gini_Index'] = zscore(country_df.Gini_Index)
country_df[(country_df.z_score_Gini_Index > 3.0)| (country_df.z_score_Gini_Index<-3.0)]


# In[125]:


country_influence = country_info.get_influence()
(c,p) = country_influence.cooks_distance


# In[126]:


plt.stem(np.arange(len(train_X)), np.round(c,3), markerfmt =",")
plt.title("Cook's Distance")
plt.xlabel("Row Index")
plt.ylabel("Cook's Distance")


# In[127]:


#Leverage Values
from statsmodels.graphics.regressionplots import influence_plot
fig, ax = plt.subplots( figsize=(8,6) ) 
influence_plot( country_info , ax = ax )
plt.title("Figure 4.4 - Leverage Value Vs Residuals") 
plt.show()

#the size of the circle is proportional to the product of residual and leverage value. The larger the circle, the larger is the residual and hence influence of the observation.
# In[128]:


pred_y = country_info.predict(test_X)


# In[129]:


from sklearn.metrics import r2_score, mean_squared_error
np.abs(r2_score(test_Y,pred_y))


# In[130]:


###################################Completed#################################################


# # Multilinear Regression (IPL dataframe)
# date:16/09/21

# In[131]:


ipl_df.columns


# In[132]:


ipl_df.head()


# In[133]:


x_features = ['AGE','COUNTRY','PLAYING ROLE','T-RUNS','T-WKTS','ODI-RUNS-S', 'ODI-SR-B', 'ODI-WKTS', 'ODI-SR-BL',
       'CAPTAINCY EXP', 'RUNS-S', 'HS', 'AVE', 'SR-B', 'SIXERS', 'RUNS-C',
       'WKTS', 'AVE-BL', 'ECON', 'SR-BL']


# In[134]:


x_features


# In[135]:


ipl_df.info()


# In[136]:


ipl_df['PLAYING ROLE'].unique()


# In[137]:


pd.get_dummies(ipl_df['PLAYING ROLE'])[0:5]


# In[138]:


categorical_features = ['AGE','COUNTRY','PLAYING ROLE','CAPTAINCY EXP']
ipl_df_encoded= pd.get_dummies(ipl_df[x_features], columns = categorical_features, drop_first=True)


# In[139]:


X_features = ipl_df_encoded.columns


# In[140]:


X_features


# In[141]:


X = sm.add_constant(ipl_df_encoded)
Y = ipl_df['SOLD PRICE']


# In[142]:


train_X,test_X,train_Y,test_Y =train_test_split(X, Y, train_size=0.8, random_state=42)


# In[143]:


train_X


# In[144]:


ipl_model= sm.OLS(train_Y, train_X).fit()
ipl_model.summary2()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




