import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
import scipy.stats as stats
from scipy.stats import ks_2samp
from scipy.stats import chisquare
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression


df= pd.read_csv('coursework_1_data_2019.csv')

## additional columns
df["environ_budget"]= df["clean_air"]+df["clean_environ"]
df["educat_budget"]= df["health_training"]+df["school_awareness"]+df["media_awareness"]+df["sub_counselling"]
df["average_environ_budget"]= (df["clean_air"]+df["clean_environ"])/10
df["average_educat_budget"]= (df["health_training"]+df["school_awareness"]+df["media_awareness"]+df["sub_counselling"])/10
df["average_increased_cases"]= (df["2018_cases_total"]-df["2008_cases_total"])/10
df[["average_environ_budget", "average_educat_budget"]] = df[["average_environ_budget", "average_educat_budget"]].astype(int)
cols= ['2008_cases_total', '2018_cases_total', 'total_budget', 'clean_air', 'clean_environ', 'school_awareness', 'media_awareness', 'sub_counselling', 'environ_budget', 'educat_budget', 'average_environ_budget', 'average_educat_budget', 'average_increased_cases', 'region']
cols = cols[-1:] + cols[:-1]
s= df.loc[:, cols]
s1= s.drop(['environ_budget', 'educat_budget', 'average_environ_budget', 'average_educat_budget', 'average_increased_cases'], axis=1, inplace=False)
s2= s[['environ_budget', 'educat_budget', 'average_environ_budget', 'average_educat_budget', 'average_increased_cases']]


## histogram
# Total environmental budget histogram (lowest=0)
plt.figure(figsize=[10,8])
n, bins, patches = plt.hist(x=s2["environ_budget"], bins='auto', color="skyblue",
                            alpha=1, rwidth=1)
plt.grid(axis='y', alpha=1)
plt.xlabel('Total Environmental Budget', size=15)
plt.ylabel('Frequency',size=15)
maxfreq = n.max()
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.axvline(s2["environ_budget"].mean(), color='k', linestyle='dashed', linewidth=1)

# After outliers
ss1=s2["environ_budget"]
ss1= [y for y in ss1 if y<402000]

plt.figure(figsize=[10,8])
n, bins, patches = plt.hist(x=ss1, bins=[50000,100000,200000,300000,400000], color="skyblue",
                            alpha=1, rwidth=1)
plt.grid(axis='y', alpha=1)
plt.xlabel('Total Environmental Budget', size=15)
plt.ylabel('Frequency',size=15)
maxfreq = n.max()
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

# Total budget histogram (lowest=2000)
plt.figure(figsize=[10,8])
n, bins, patches = plt.hist(x=s["total_budget"], bins=[50000,100000,200000,300000,400000,500000,600000,700000,800000,900000], color="skyblue",
                            alpha=1, rwidth=1)
plt.grid(axis='y', alpha=1)
plt.xlabel('Total Budget', size=15)
plt.ylabel('Frequency',size=15)
maxfreq = n.max()
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.axvline(s["total_budget"].mean(), color='k', linestyle='dashed', linewidth=1)

# After outliers
ss2=s["total_budget"]
ss2= [y for y in ss2 if y<1010000 and y>99000]

plt.figure(figsize=[10,8])
n, bins, patches = plt.hist(x=ss2, bins=[100000,200000,300000,400000,500000,600000,700000,800000,900000], color="skyblue",
                            alpha=1, rwidth=1)
plt.grid(axis='y', alpha=1)
plt.xlabel('Total Budget', size=15)
plt.ylabel('Frequency',size=15)
maxfreq = n.max()
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


# Comparing two histograms
plt.figure(figsize=[10,8])
plt.hist(s2["environ_budget"], bins=[50000,100000,200000,300000,400000,500000,600000,700000,800000,900000], alpha=0.5, label='Environmental Budget')
plt.hist(s["total_budget"], bins=[50000,100000,200000,300000,400000,500000,600000,700000,800000,900000], alpha=0.5, label='Total Budget')
plt.legend(loc='upper right')
plt.legend(fontsize=15)
plt.xlabel('Budget', size=15)
plt.ylabel('Frequency',size=15)
plt.show()

# After outliers
plt.figure(figsize=[10,8])
plt.hist(ss1, bins=[50000,100000,200000,300000,400000,500000,600000,700000,800000,900000], alpha=0.5, label='Environmental Budget')
plt.hist(ss2, bins=[50000,100000,200000,300000,400000,500000,600000,700000,800000,900000], alpha=0.5, label='Total Budget')
plt.legend(loc='upper right')
plt.legend(fontsize=15)
plt.xlabel('Budget', size=15)
plt.ylabel('Frequency',size=15)
plt.show()

## Chi-Square
crosstab = pd.crosstab(s2["environ_budget"], s["total_budget"])
chisquare(crosstab,axis=None)


## Spearman's Rank Correlation
coef, p = spearmanr(df["average_environ_budget"],df["average_increased_cases"])
print('Spearmans correlation coefficient: %.3f' % coef)
alpha = 0.05
if p > alpha:
    print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
else:
    print('Samples are correlated (reject H0) p=%.3f' % p)

## Single Regression
x= np.array(df["environ_budget"]).reshape((-1,1))
y= df["2018_cases_total"]
model = LinearRegression().fit(x, y)

r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

# Regression plot
y_pred = model.intercept_ + model.coef_ * x

plt.figure(figsize=[8,6])
plt.scatter(x, y,  color='m')
plt.plot(x, y_pred, color='g', linewidth=3)

plt.xlabel('Average Increased Cases', size=15) 
plt.ylabel('Average Environmental Budget', size=15) 

plt.show()