#!/usr/bin/env python
# coding: utf-8

# # Machine Learning

# In[7]:


import sys
sys.path.insert(0,'..')
from datastore import DataStore
from ml import Learner


# Initialize data store object, then learner, automatically loading feature file produced by featurizer, along with file of data labels, and merging features to labels.

# In[2]:


datastore = DataStore(cfg_dir='../configs/config.yml')
learner = Learner(datastore=datastore, clean_folders=True)


# Experiment quickly with untuned models to get a sense of accuracy. Lasso, Ridge, random forest, and gradient boosting models are implemented natively, other models can be implemented by hand.

# In[3]:


lasso_scores = learner.untuned_model('lasso')
randomforest_scores = learner.untuned_model('randomforest')
print('LASSO', lasso_scores)
print('Random Forest', randomforest_scores)


# Fine-tune a gradient boosting model, tuning hyperparameters over cross validation, and produce predictions for all labeled observations out-of-sample over cross-validation. Also generate predictions for all subscribers in the feature dataset. 

# In[4]:


gradientboosting_scores = learner.tuned_model('gradientboosting')
print('Gradient Boosting (Tuned)', gradientboosting_scores)
learner.oos_predictions('gradientboosting', kind='tuned')
learner.population_predictions('gradientboosting', kind='tuned')


# Evaluate the model’s accuracy. Produce a scatterplot of true vs. predicted values with a LOESS fit and a bar plot of the most important features. Generate a table showing the targeting accuracy, precision, and recall of the predictions for nine hypothetical targeting scenarios (targeting between 10% and 90% of the population). 

# In[6]:


learner.scatter_plot('gradientboosting', kind='tuned')
learner.feature_importances_plot('gradientboosting', kind='tuned')
learner.targeting_table('gradientboosting', kind='tuned')

