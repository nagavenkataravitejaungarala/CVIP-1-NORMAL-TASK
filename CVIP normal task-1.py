#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split


# In[55]:


data =pd.read_csv('covid19.csv')


# In[56]:


data


# In[57]:


#HEAD it shows first 5


# In[58]:


data.head()


# In[59]:


# tail it shows last 5


# In[60]:


data.tail()


# In[61]:


# Data cleaning


# In[62]:


# Drop unnecessary columns
columns_to_drop = ['Active Ratio']  
data = data.drop(columns=columns_to_drop)
# Rename columns
new_column_names = {'Population': 'Populations'}  
data = data.rename(columns=new_column_names)
# Remove duplicate rows
data = data.drop_duplicates()
# Handle missing values
data = data.dropna()  
# Save the cleaned data to a new CSV file
data.to_csv('cleaned_covid_data.csv', index=False)


# In[63]:


data


# In[64]:


#data preprocessing


# In[65]:



# Normalize numerical features
numerical_columns = ['Active', 'Discharged']  
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

X = data.drop('State/UTs', axis=1)  
y = data['Discharge Ratio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.to_csv('preprocessed_X_train.csv', index=False)
X_test.to_csv('preprocessed_X_test.csv', index=False)
y_train.to_csv('preprocessed_y_train.csv', index=False)
y_test.to_csv('preprocessed_y_test.csv', index=False)


# In[66]:


X


# In[67]:


y


# In[68]:


data[numerical_columns]


# In[69]:


#exploratory data analysis


# In[70]:


print(data.describe())

print(data.dtypes)

print(data.isnull().sum())


# In[71]:


# Plot the distribution of a numerical variable
plt.figure(figsize=(10, 6))
sns.histplot(data['Discharged'], kde=True)
plt.title('Distribution of Discharged')
plt.xlabel('Discharged')
plt.ylabel('Count')
plt.show()


# In[72]:


# Plot the relationship between two numerical variables
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Active', y='Discharged', data=data)
plt.title('Relationship between Active and Discharged')
plt.xlabel('Discharged')
plt.ylabel('Active')
plt.show()


# In[73]:


# Plot a bar chart of categorical variable counts
plt.figure(figsize=(10, 6))
sns.countplot(x='Death Ratio', data=data)
plt.title('Death Ratio Counts')
plt.xlabel('Death Ratio')
plt.ylabel('Count')
plt.show()



# In[74]:


# Plot a heatmap of correlation between numerical variables
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Perform other exploratory data analysis as per your requirements
# Such as box plots, violin plots, pair plots, etc.


# In[75]:


#data aggregration and grouping


# In[76]:


grouped_data = data.groupby('Deaths').mean()
grouped_data = data.groupby(['Active', 'Deaths']).sum()['Discharged']

grouped_data = grouped_data.reset_index()

grouped_data.to_csv('aggregated_covid_data.csv', index=False)


# In[77]:


grouped_data


# In[78]:


#stastical analyis


# In[79]:



mean_value = data['Active'].mean()

median_value = data['Deaths'].median()

mode_value = data['Discharged'].mode()[0]

std_deviation = data['Death Ratio'].std()

group1 = data[data['Active'] == 'Group1']['Deaths']
group2 = data[data['Discharge Ratio'] == 'Group2']['Death Ratio']
t_statistic, p_value = stats.ttest_ind(group1, group2)

contingency_table = pd.crosstab(data['Discharge Ratio'], data['Death Ratio'])
chi2_statistic, p_value, _, _ = stats.chi2_contingency(contingency_table)
# Print the results
print("Mean:", mean_value)
print("Median:", median_value)
print("Mode:", mode_value)
print("Standard Deviation:", std_deviation)
print("T-Statistic:", t_statistic)
print("P-Value:", p_value)
print("Chi-Square Statistic:", chi2_statistic)
print("P-Value:", p_value)

