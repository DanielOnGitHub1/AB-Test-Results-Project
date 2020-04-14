#!/usr/bin/env python
# coding: utf-8

# # <font color='blue'>A new webpage's implementation study through A/B Test Results</font>
# 
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Preparing our Dataframe](#probability)
# - [A/B Testing](#ab_test)
# - [A Regression Approach](#regression)
# 
# 
# <a id='intro'></a>
# ## Introduction
# 
# 
# An e-commerce website wants to decide whether or not should they change their webpage to a new one, in order to increase their conversion rate. For this project, we will be working to understand the results of an A/B test, with the objective of helping the company to understand if they should implement the new page or keep the old page, based in the conversion rates observed through a control group, which was presented with the old page, and a treatment group, presented with the new page.
# 
# 
# <a id='probability'></a>
# ### Preparing our Dataframe
# 
# To get started, let's import our libraries.

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# Now, let's read the csv file in Jupyter Notebook, and take an overall look in the created dataframe:

# In[2]:


df = pd.read_csv('ab_data.csv')
df.head()


# The time horizon of the available user data:

# In[3]:


print('First row date: {}'.format(df.timestamp.min()))
print('Last row date: {}'.format(df.timestamp.max()))


# The number of rows in the dataset:

# In[4]:


print('Number of rows: {}'.format(df.shape[0]))


# The number of unique users in the dataset:

# In[5]:


print('Number of unique users: {}'.format(df.user_id.nunique()))


# The proportion of users converted:

# In[6]:


conv_users = df.query('converted == 1').shape[0]
print('Proportion of converted users: {}'.format(conv_users/df.shape[0]))


# We have to verify the cases in which the landing page doesn't correctly identify to the group. The new landing page should be related to the treatment group, and the old page with the control group. This is the number of incorrect matches:

# In[7]:


wrong_match = df[((df.group == "treatment") & (df.landing_page != "new_page")) | 
   ((df.group == "control") & (df.landing_page == "new_page"))].shape[0]
print("incorrect matches: {}".format(wrong_match))


# Verifying if any of the rows have missing values:

# In[8]:


df.isna().sum()
print('total of missing values: \n{}'.format(df.isnull().sum()))


# For the rows where **treatment** does not match with **new_page** or **control** does not match with **old_page**, we cannot be sure if this row truly received the new or old page.  With that in mind, let's remove these unnecessary rows:

# In[9]:


df2 = df[((df.group == "treatment") & (df.landing_page == "new_page")) | ((df.group == "control") & (df.landing_page == "old_page"))]
df2.shape


# In[10]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# After it, this is the current number of unique users:

# In[11]:


df2.user_id.nunique()


# Comparing this result with the shape of the new df2, which is 290585 for the number of rows, we can see that there's one user that is not unique, meaning that there's a duplicated user_id in 2 rows. Let's find this user_id:

# In[12]:


df2.user_id.value_counts()[:1]


# In[13]:


df2.query('user_id == 773192')


# After identifying the user_id and its respective rows, let's remove one of them:

# In[14]:


df2.drop([1899], axis=0, inplace=True)


# In[15]:


df2.shape


# ### Conversion Rate analysis

# With our dataframe composed only by unique users and the correct relations between landing page and group, let's check our conversion rate. first, let's analyze the conversion without distinguishment between pages or groups and pose some questions about it:

# In[16]:


conv_all = df2.converted.mean()
print('probability of converting: {}'.format(conv_all))


# Given that an individual was in the **control** group, what is the probability they converted?

# In[17]:


conv_control = df2.query('group == "control"').converted.mean()
print('probability of converting in the control group: {}'.format(conv_control))


# Given that an individual was in the **treatment** group, what is the probability they converted?

# In[18]:


conv_treat = df2.query('group == "treatment"').converted.mean()
print('probability of converting in the treatment group: {}'.format(conv_treat))


# What is the probability that an individual received the new page?

# In[19]:


receive_new = df2.query('landing_page == "new_page"').shape[0]/df2.shape[0]
print('probability of receiving the new page: {}'.format(receive_new))


# **According to the results, the probability of conversion in the new page (treatment group) is smaller than in the old page (control group). Furthermore, the number of individuals in each group doesn't imply any considerable alteration in the probabilities, since both group sizes are very similar.**

# <a id='ab_test'></a>
# ## A/B Testing
# 
# Based on all the data provided, and assuming that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, let's define our null and alternative hypothesisin terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# _H0 = P old >= P new_ <br>
# _H1 = P old < P new_

# Under the null hypothesis, we assume that both $p_{new}$ and $p_{old}$ have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, let's assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br>
# 
# with that in mind, let's use a sample size for each page equal to the ones in **ab_data.csv**, and perform a sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null:

# a. What is the **conversion rate** for $p_{new}$ under the null? 

# In[20]:


p_new = df.converted.mean()
p_new


# b. What is the **conversion rate** for $p_{old}$ under the null? <br><br>

# In[21]:


p_old = df.converted.mean()
p_old


# c. What is $n_{new}$, the number of individuals in the treatment group?

# In[22]:


df.query('group == "treatment"').shape[0]


# d. What is $n_{old}$, the number of individuals in the control group?

# In[23]:


df.query('group == "control"').shape[0]


# Now let's simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null, and create a new variable **new_page_converted** to store these $n_{new}$ 1's and 0's in.

# In[24]:


new_page_converted = []

for x in range(147275):
    result = np.random.choice([0,1], p= [0.8804, 0.1196])
    new_page_converted.append(result)


# now with $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null, to store the result in **old_page_converted**.

# In[25]:


old_page_converted = []

for x in range(147202):
    result = np.random.choice([0,1], p= [0.8804, 0.1196])
    old_page_converted.append(result)


# We are now able to check the differentes between their conversion rates, by calculation our previous results' means and subtracting them:

# In[26]:


diff_p = np.array(new_page_converted).mean() - np.array(old_page_converted).mean()
diff_p


# It is observable that the old page got a higher conversion success result concerning these 10.000 iterations... But what about if we ran the same simulation for checking the success rate difference 10.000 times?

# In[27]:


p_diffs = []

for x in range(10000):
    p_new = np.random.choice([0,1], size= 147275, p= [0.8804, 0.1196]).mean()
    p_old = np.random.choice([0,1], size= 147202, p= [0.8804, 0.1196]).mean()
    p_diffs.append(p_new - p_old)


# Here's a histogram to represent the results:

# In[28]:


plt.hist(p_diffs);


# Below is the proportion of the **p_diffs** that are greater than the actual difference observed in **ab_data.csv**:

# In[29]:


p_diff_df2 = df2.query('group=="treatment"')['converted'].value_counts(normalize=True)[1] - df2.query('group=="control"')['converted'].value_counts(normalize=True)[1]
p_diff_df2


# In[30]:


(p_diffs > p_diff_df2).mean()


# ### Approximately 90% of the proportion differences are greater that the actual difference in ab_data.csv, leading us to fail to reject the null hypothesis. Therefore, it indicates us to maintain the old page.

# Here's another approach for the conversion comparison, made through the use of a built-in: 

# In[31]:


import statsmodels.api as sm

convert_old = df2.query('converted == 1 & landing_page == "old_page"').shape[0]
convert_new = df2.query('converted == 1 & landing_page == "new_page"').shape[0]
n_old = 147202
n_new = 147275


# In[32]:


z_score, p_value = sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new], alternative='smaller')
z_score, p_value


# **Since we can observe a Z-score value of 1.33, it tells us that the results are 1.33 standard deviations away from the mean. In parallel, a P-value of approximately 0.9 pushes us away from the null hypothesis. Also, A positive Z-score of 1.33 tells us that the distribution is not unusual/exceptional. In conclusion, we can observe that the conversion rates are higher for the old page.**

# <a id='regression'></a>
# ## A regression approach

# ### Logistic Regression

# Through the use of a logistic regression model, let's see if there is a significant difference in conversion based on which page a customer receives. For that, we will create a dummy variable column for which page each user received, an **intercept** column and an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[33]:


df2['intercept'] = 1
df2[['new_page','old_page']] = pd.get_dummies(df2['landing_page'])
df2[['control','ab_page']] = pd.get_dummies(df2['group'])
df2.head()


# Now we will fit our model to identify if there's a significant relation between the new page and the conversion:

# In[34]:


import statsmodels.api as sm
model = sm.Logit(df2['converted'], df2[['intercept','ab_page']])
result = model.fit()


# This is the model's summary:

# In[35]:


result.summary()


# **The p-value of ab_page is 0.19, which is beyond the alpha threshold. It differs from the p-value in the previous test because in part II the null hypothesis was defined as  "P old >= p new", while in part III the null was defined as "P old = P new". With that in mind, the ab_page variable in part III has a smaller P-value, due to the value amplitude being larger on the logistic regression (two-tailed).

# **A correlation analysis can be drastically altered with the inclusion (or exclusion) of exogenous variables. If you choose to insert other factors into your model, you'll be given the chance to unravel insighthful discoveries, which will grant you a better comprehension about the dependent variable, and it's a good practice to adopt different angles and perspectives during your EDA. On the other hand, you should be careful with certain aspects such as the rise of false positives, intercollinearity risks and other multiple comparison issues, which must be addressed with different methods depending on the scenario.**

# ### To check the exogenous variables' influence, let's input the country variable, to observe if there's any relevant alteration in the regression model's results, extracting it from another file:

# In[46]:


df_country = pd.read_csv('countries.csv')
df_country.head()


# In[47]:


df3 = df2.set_index('user_id').join(df_country.set_index('user_id'))
df3.head(8)


# In[48]:


df3[['CA','UK','US']] = pd.get_dummies(df3.country)
df3.head()


# Let's check the dependent and independent variable's correlation, with **country** being the only exogenous variable:

# In[49]:


model = sm.Logit(df3['converted'], df3[['intercept', 'UK', 'US']] )
result = model.fit()
result.summary()


# **By checking the results, if we consider the alpha threshold at 0.05, there is not a significance between country and conversion.**

# now we'll look whether an interaction between **ab_page** and **country** has a significant effect on conversion:

# In[79]:


model = sm.Logit(df3['converted'], df3[['intercept', 'ab_page', 'UK', 'US']])
result = model.fit()
result.summary()


# **Again, considering a threshold of alpha at 0.05, we do not have significance between the interation among countries and pages to the conversion rate**

# <a id='conclusions'></a>
# ## Conclusion <br>
# 
# After analyzing the data provided, it is plausible to say that the deployment of a new web page will not necessarily accrue higher conversion rates. Through the hypothesis testing applied in this project, we fail to reject the null, demonstrating technical evidence of no significant relation between the variables. However, this study utilizes a database containing observations made in a shorter than 1 month period, which brings us the question if these observations should persist for longer, concerning intrinsic issues such as the Novelty Effect or change aversion.
# For now, with the data at hand the e-commerce website should persist in using the current web page, giving them the opportunity to search for different optimizations or changes to try to boost their conversion numbers.
# 

# In[48]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])


# In[ ]:




