############ Hands on ML book. Chapter 2###############################

##### Steps ######

#1. Looking at the picture
#2. Get the data
#3. Discover and visualise the data
#4. Prepare the data for Machine Learning algorithm
#5. Select model and train it
#6. Fine-tune the model
#7. Present solution
#8. Launch, monitor and maintain the system

#######################################################################
#%%

#For this the data set will be the California Housing Prices dataset

# Function to download the data and decompress

import os
import tarfile
from six.moves import urllib

dl_root = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
#dl_root = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/'
house_path = 'datasets/housing'
house_url = dl_root + house_path + '/housing.tgz'

def get_housing_data(housing_url=house_url, housing_path=house_path):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url,tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
#Creating the dataframe with all the .csv housing data

import pandas as pd

def load_housing_data(housing_path=house_path):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)
#%%
    
get_housing_data()
housing_data = load_housing_data()

#Exploring the data
pd.set_option('display.max_columns', None)
print(housing_data.head())

#Checking for missing data
print(housing_data.info())
print(housing_data.isnull().sum())
print(housing_data.isna().sum())

#Total_bedrooms has 207 missing data points

cat_data_view = housing_data['ocean_proximity'].value_counts()
print(cat_data_view)
housing_stats = housing_data.describe()
print(housing_stats)

#Visualise the data
housing_data.hist(bins=50, figsize=(20,15))

#Weird dist. Some attributes might be capped, meaning that there is a sudden jump in count at the max value.
#Also alot of them are tail heavy.

#%%
#########################Create a test subset############################

#Using sklearn as a purely random separation

from sklearn.model_selection import train_test_split

housing_data_train, housing_data_test = train_test_split(housing_data, test_size=0.2, random_state=42) 

# This is fine and it separates the sets consistently and it hold as the data set is updated, avoiding the snooping bias.
# Use a stratified sampling. We discovered through asking experts that income is important in predicting prices
# so we will use it to stratify the data.
#%%

import numpy as np

housing_data['income_categorical'] = np.ceil(housing_data['median_income']/1.5)
housing_data['income_categorical'].where(housing_data['income_categorical']<5,5.0,inplace=True)

housing_data.hist(bins=50, figsize=(20,15))

from sklearn.model_selection import StratifiedShuffleSplit

Split_indices_mod = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

#With for loop
'''for train_index, test_index in Split_indices_mod.split(housing_data, housing_data['income_categorical']): 
    strat_train_set = housing_data.loc[train_index]
    strat_test_set = housing_data.loc[test_index]
'''
#Without for loop. The comma in the lhs is there to unpack the tuple which is the output from the .split() method
(train_index, test_index), = Split_indices_mod.split(housing_data, housing_data['income_categorical'])
strat_train_set = housing_data.loc[train_index]
strat_test_set = housing_data.loc[test_index]

#%%

#Checking the proportions
table_comparison = pd.concat([housing_data['income_categorical'].value_counts()/len(housing_data),
                              strat_train_set['income_categorical'].value_counts()/len(strat_train_set),
                              strat_test_set['income_categorical'].value_counts()/len(strat_test_set)],axis=1)

table_comparison.columns = ['Original dataset','Train dataset','Test dataset']
#print(table_comparison)

#Getting rid of the categorical variable so that the data goes back to the original state
strat_train_set.drop(['income_categorical'],axis=1,inplace=True)
strat_test_set.drop(['income_categorical'],axis=1,inplace=True)
#%%

#####################################Discover and visualise the data to get insight##############################

import matplotlib.pyplot as plt

play_around_set = strat_train_set.copy()

#Include the subplot command to bypass the pandas bug that prvents the X-axis labe to show
fig, ax = plt.subplots()
play_around_set.plot(kind = 'scatter', x = 'longitude', y = 'latitude', alpha = 0.1, s=play_around_set['population']/100,
                     label = 'population',c='median_house_value',colormap = 'jet',colorbar = True,ax=ax)

# Taken from the book

'''
This image tells you that the housing prices are very much related to the location
(e.g., close to the ocean) and to the population density, as you probably knew already.
It will probably be useful to use a clustering algorithm to detect the main clusters, and
add new features that measure the proximity to the cluster centers. The ocean prox‐
imity attribute may be useful as well, although in Northern California the housing
prices in coastal districts are not too high, so it is not a simple rule.
'''

# I do not know if I agree about the population being relevant but we'll see.

#%%
##########################Looking for correlations###############################

correlation_matrx = play_around_set.corr()
print(correlation_matrx['median_house_value'].sort_values(ascending=False))

#Just picking some interesting attributes

from pandas.plotting import scatter_matrix

att_of_interest=['median_house_value','median_income','total_rooms','housing_median_age']
scatter_matrix(play_around_set[att_of_interest],alpha=0.5,figsize=(12,8),marker='o',edgecolor='black',color='#1f77b4',
               linewidths=0.5, hist_kwds={'linewidth':'1','edgecolor':'black'})


#Focusing on median house income
fig, ax = plt.subplots()
play_around_set.plot(kind = 'scatter', x = 'median_income', y = 'median_house_value', alpha = 0.5,
                     ax=ax,marker='o',edgecolor='black',color='#1f77b4',linewidths=0.5,s=50)

#Beware the horizontal lines. Don't want to learn those...probably.
#%%
###########################Experimenting with attribute combos####################

'''
You also noticed that some attributes have a tail-heavy distribution, so you may want to trans‐
form them (e.g., by computing their logarithm).
'''

play_around_set["rooms_per_household"] = play_around_set["total_rooms"]/play_around_set["households"]
play_around_set["bedrooms_per_room"] = play_around_set["total_bedrooms"]/play_around_set["total_rooms"]
play_around_set["population_per_household"] = play_around_set["population"]/play_around_set["households"]

correlation_matrx = play_around_set.corr()
print(correlation_matrx['median_house_value'].sort_values(ascending=False))

play_around_set.plot(kind = 'scatter', x = 'bedrooms_per_room', y = 'median_house_value', alpha = 0.5,
                     marker='o',edgecolor='black',color='#1f77b4',linewidths=0.5,s=50)
plt.axis([0, 1, 0, 520000])

#Bedrooms per roomis more correlated with median value than the attributes we use to create it, 
# so it is a good addition to the set

#%%
#################################Preping the data for ML algorithms#############################

housing = strat_train_set.drop('median_house_value',axis=1,inplace=False)
housing_labels = strat_train_set['median_house_value'].copy()

#%%
#Data Cleaning wo scikit learn

#Remember that total_bedrooms has missing data. What to do?
'''
• Get rid of the corresponding districts (rows).
• Get rid of the whole attribute.
• Set the values to some value (zero, the mean, the median, etc.).
'''

housing.dropna(subset=['total_bedrooms']) #Option 1
housing.drop('total_bedrooms',axis=1,inplace=True)  #Option 2
median=housing['total_bedrooms'].median()  #Option 3
housing['total_bedrooms'].fillna(median)

'''
If you choose option 3, you should compute the median value on the training set, and
use it to fill the missing values in the training set, but also don’t forget to save the
median value that you have computed. You will need it later to replace missing values
in the test set when you want to evaluate your system, and also once the system goes
live to replace missing values in new data.
'''
#%%

#Data Cleaning with scikit learn. We can use imputer

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy = 'median')

#We need a copy of the set without the variable ocean proximity because imputer can only wirk with numnerical values

housing_numeric = housing.drop('ocean_proximity',axis=1,inplace=False)
imputer.fit(housing_numeric)
print(imputer.statistics_,housing_numeric.median().values)

#Now transforming the data, meaning, getting the data with the na replaced by the median

X = imputer.transform(housing_numeric)
housing_tr = pd.DataFrame(X,columns=housing_numeric.columns) 

#%%
################### Handling text and categoricals################################

#Turning the ocean_proximity from text into numbers so we can work with it

'''
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
housing_cat = housing['ocean_proximity']
housing_cat_encoded = encoder.fit_transform(housing_cat)
print(encoder.classes_)
'''
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()
housing_cat = housing[['ocean_proximity']]
housing_cat_encoded = encoder.fit_transform(housing_cat)
print(encoder.categories_)

'''
One issue with this representation is that ML algorithms will assume that two nearby
values are more similar than two distant values. Obviously this is not the case (for
example, categories 0 and 4 are more similar than categories 0 and 1). To fix this
issue, a common solution is to create one binary attribute per category: one attribute
equal to 1 when the category is “<1H OCEAN” (and 0 otherwise), another attribute
equal to 1 when the category is “INLAND” (and 0 otherwise), and so on. This is
called one-hot encoding, because only one attribute will be equal to 1 (hot), while the
others will be 0 (cold).
'''

from sklearn.preprocessing import OneHotEncoder

encoder=OneHotEncoder()
housing_cat_onehot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))

#Can convert it to an array if want to but if there are too many categories it would be very memory intensive.

#housing_cat_onehot_array = housing_cat_onehot.toarray()

#We can do both these steps with the onehotencoder now. If we use this, then the 
#dataframe or series has to be in a (x,1) shape.
#%%
encoder=OneHotEncoder()
#In case we want an array
#encoder = OneHotEncoder(sparse=False)
housing_cat_onehot = encoder.fit_transform(housing_cat)

print(encoder.categories_)
#%%
######################Custom Transformers######################################

'''
Although Scikit-Learn provides many useful transformers, you will need to write
your own for tasks such as custom cleanup operations or combining specific
attributes. You will want your transformer to work seamlessly with Scikit-Learn func‐
tionalities (such as pipelines), and since Scikit-Learn relies on duck typing (not inher‐
itance), all you need is to create a class and implement three methods: fit()
(returning self ), transform() , and fit_transform() . You can get the last one for
free by simply adding TransformerMixin as a base class. Also, if you add BaseEstima
tor as a base class (and avoid *args and **kargs in your constructor) you will get
two extra methods ( get_params() and set_params() ) that will be useful for auto‐
matic hyperparameter tuning. For example, here is a small transformer class that adds
the combined attributes we discussed earlier:
'''

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = [
    list(housing.columns).index(col)
    for col in ("total_rooms", "total_bedrooms", "population", "households")]

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
#%%

#Second option to do the same sort of

from sklearn.preprocessing import FunctionTransformer

rooms_ix, bedrooms_ix, population_ix, household_ix = [
    list(housing.columns).index(col)
    for col in ("total_rooms", "total_bedrooms", "population", "households")]

def add_extra_features(X, add_bedrooms_per_room=True):
    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]
    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household,
                     bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]


attr_adder = FunctionTransformer(add_extra_features, validate=False,
                                 kw_args={"add_bedrooms_per_room": False})
housing_extra_attribs = attr_adder.fit_transform(housing.values)
#%%
#############################Feature scaling##################################

'''
We can min-max or standarise:
    
    - min-max puts everythin in a 0-1 value range (good for neural networks)
      but it is very suceptible to outliers.
    - standarise puts everything in a zero-mean and one-variance...not as good for some
      algorithms but resistant to outliers.
'''

# Transformation pipelines

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

cat_att = ['ocean_proximity']
num_att = list(housing_numeric)

numerical_transformer = Pipeline([
    ('imputer',SimpleImputer(strategy = 'median')),
    ('adder',FunctionTransformer(add_extra_features, validate=False)),
    ('stadariser',StandardScaler())])

categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer([
    ('numeric',numerical_transformer,num_att),
    ('categorical',categorical_transformer,cat_att)])

housing_ready = preprocessor.fit_transform(housing)
print(housing_ready.shape)
#%%
########################Select and train a model#############################

#Training linear regression models

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_ready,housing_labels) 

#Checking with some data from training set

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_ready = preprocessor.transform(some_data)
print('Predictions: \t',lin_reg.predict(some_data_ready))
print('Labels: \t',list(some_labels))

#not great...lets check the RMSE

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_ready)
lin_mse = mean_squared_error(housing_labels,housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print('RMSE:', lin_rmse)

'''
Okay, this is better than nothing but clearly not a great score: most districts’
median_housing_values range between $120,000 and $265,000, so a typical predic‐
tion error of $68,628 is not very satisfying.

This is underfitting. How do we fix that? select a more powerful model, to feed 
the training algorithm with better features, or to reduce the constraints on the model.
'''
#%%
#Let's try a decision tree model

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_ready,housing_labels)

#Evaluate on the training set

tree_reg_prediction = tree_reg.predict(housing_ready)
tree_mse = mean_squared_error(housing_labels,tree_reg_prediction)
tree_rmse = np.sqrt(tree_mse)
print('RMSE:', tree_rmse)

#The result is 0...which means that we probably overfitted the data quite badly.
#We'll have to try cross-validation so that we can try to address these issues wo
#having to look at the test set
#%%
#################Cross-validation###########################

#We'll divide the training set into 10 to train the model 10 times and validate it all within the
#training set

from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_ready, housing_labels, scoring = 'neg_mean_squared_error', cv = 10)
rmse_scores = np.sqrt(-scores)
#Trying the build-in metric
scores_2 = cross_val_score(tree_reg, housing_ready, housing_labels, scoring = 'neg_root_mean_squared_error', cv = 10)
#They are different enough
print(np.allclose(rmse_scores,scores_2,rtol = 1.e+1))

#%%
#Printing their stats

def cross_val_stats(scores):
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standad Deviation:', scores.std())
    
cross_val_stats(rmse_scores)

'''
The Decision Tree model is overfitting so badly that it performs worse
than the Linear Regression model
'''    

#%%

#Trying the random forst regressor

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_ready,housing_labels)
forest_reg_prediction = forest_reg.predict(housing_ready)
forest_reg_mse = mean_squared_error(housing_labels,forest_reg_prediction)
forest_reg_rmse = np.sqrt(forest_reg_mse)
print('RMSE:', forest_reg_rmse)

#Now with cross validation
#%%
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_ready, housing_labels, scoring = 'neg_mean_squared_error', cv = 10)
rmse_forest_scores = np.sqrt(-forest_scores)

cross_val_stats(rmse_forest_scores)

#%%
#Saving everything

from joblib import dump,load

dump(forest_reg,'forest_model.pkl')
#later
forest_model = load('forest_model.pkl')
#%%
##########################Fine-tuning the model###########################

#Tunning hyperparameters with GridSearchCV
from sklearn.model_selection import GridSearchCV

parameter_grid = [
    {'n_estimators':[10,50,100],'max_features':[2,4,6,8]},
    {'bootstrap':[False],'n_estimators':[10,50,100],'max_features':[2,3,4]}]

forest_reg_grid = GridSearchCV(RandomForestRegressor(), parameter_grid, cv = 5, scoring='neg_mean_squared_error',refit=True)
forest_reg_grid.fit(housing_ready,housing_labels)
#%%


print('Best_estimator:',forest_reg_grid.best_estimator_)
print('Best params:', forest_reg_grid.best_params_)
print('Best_score_rmse:',np.sqrt(-forest_reg_grid.best_score_))

cvres = forest_reg_grid.cv_results_
for score, params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-score),params)
#%%
#Tunning hyperparameters with RandomSearch

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg_random = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_distribs, 
                                     n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42,refit=True)
forest_reg_random.fit(housing_ready,housing_labels)
#%%
print('Best_estimator:',forest_reg_random.best_estimator_)
print('Best_score_rmse:',np.sqrt(-forest_reg_random.best_score_))

cvres = forest_reg_random.cv_results_
for score, params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-score),params)
    
#%%
#Check the feature's importances
feature_importances = forest_reg_grid.best_estimator_.feature_importances_
#print(feature_importances)

extra_atts = ['rooms_per_household','population_per_household','bedrooms_per_room']
cat_atts = list(preprocessor.named_transformers_['categorical'].categories_[0])
num_atts = list(housing_numeric)

All_atts = num_atts + extra_atts + cat_atts
list_of_importances = sorted(zip(feature_importances, All_atts),reverse=True)
for i in list_of_importances:
    print(i)
    
'''
With this information, you may want to try dropping some of the less useful features
(e.g., apparently only one ocean_proximity category is really useful, so you could try
dropping the others).
You should also look at the specific errors that your system makes, then try to under‐
stand why it makes them and what could fix the problem (adding extra features or, on
the contrary, getting rid of uninformative ones, cleaning up outliers, etc.).
'''

#%%

###############################Evaluate on the test set#####################################

final_model = forest_reg_grid.best_estimator_


test_set = strat_test_set.drop('median_house_value',axis=1,inplace=False)
test_labels = strat_test_set['median_house_value'].copy()
test_ready = preprocessor.transform(test_set)

final_predictions = final_model.predict(test_ready)
final_rmse = np.sqrt(mean_squared_error(test_labels,final_predictions))
print('The final RMSE for the test set:', final_rmse)

#%%
##############################Excercises#############################################

'''
Ex. 1

Try a Support Vector Machine regressor ( sklearn.svm.SVR ), with various hyper‐
parameters such as kernel="linear" (with various values for the C hyperpara‐
meter) or kernel="rbf" (with various values for the C and gamma
hyperparameters). Don’t worry about what these hyperparameters mean for now.
How does the best SVR predictor perform?
'''

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
parameter_grid_SVR = [
        {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]

SVR_grid = GridSearchCV(SVR(), parameter_grid_SVR, cv = 5, scoring='neg_mean_squared_error', verbose=2, n_jobs=4)
SVR_grid.fit(housing_ready,housing_labels)
#%%


print('Best_estimator:',SVR_grid.best_estimator_)
print('Best_score_rmse:',np.sqrt(-SVR_grid.best_score_))

cvres = SVR_grid.cv_results_
for score, params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-score),params)
    
#%%
'''
Ex 2

Try replacing GridSearchCV with RandomizedSearchCV .
'''
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV


param_distribs_random = {'kernel': ['linear','rbf'],
        'C': randint(low=0, high=200000),
        'gamma': randint(low=0.001, high=8),
    }

SVR_grid_ran = RandomizedSearchCV(SVR(), param_distribs_random, 
                                     n_iter=50, cv=5, scoring='neg_mean_squared_error', 
                                     random_state=42,verbose=2, n_jobs=4)
SVR_grid_ran.fit(housing_ready,housing_labels)

print('Best_estimator:',SVR_grid_ran.best_estimator_)
print('Best_score_rmse:',np.sqrt(-SVR_grid_ran.best_score_))

cvres = SVR_grid_ran.cv_results_
for score, params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-score),params)   
#%%
'''
Ex 3

Try adding a transformer in the preparation pipeline to select only the most
important attributes.
'''

k_k = 8

def Top_features(X,k):
    return X[:,np.argpartition(feature_importances, -k)[-k:]]
    

Pipeline_with_selection = Pipeline([
    ('preprocessor', preprocessor),
    ('selector', FunctionTransformer(Top_features, validate=False,kw_args={"k": k_k}))])
    


housing_select_ready = Pipeline_with_selection.fit_transform(housing)

#Checking

print(housing_select_ready[0:3])
print(housing_ready[0:3,np.argpartition(feature_importances, -k_k)[-k_k:]])

#%%
'''
Ex 4

Try creating a single pipeline that does the full data preparation plus the final
prediction.
'''

from sklearn.ensemble import RandomForestRegressor


Pipeline_full = Pipeline([
    #('preparation with no selection', preprocessor),
    ('preparation_with_selection', Pipeline_with_selection),
    ('model_fit_grid_search', RandomForestRegressor()) 
    #('model fit grid search', RandomForestRegressor(**forest_reg_grid.best_params_)) #Its RMSE =  0 with this
    ])

Forest_reg_full_pipe = Pipeline_full.fit(housing,housing_labels)

#%%
Forest_reg_full_pipeprediction = Forest_reg_full_pipe.predict(housing)
Forest_reg_full_pipemse = mean_squared_error(housing_labels,Forest_reg_full_pipeprediction)
Forest_reg_full_pipermse = np.sqrt(Forest_reg_full_pipemse)
print('RMSE:', Forest_reg_full_pipermse)

some_data = housing.iloc[:4]
some_labels = housing_labels.iloc[:4]

print("Predictions:\t", Forest_reg_full_pipe.predict(some_data))
print("Labels:\t\t", list(some_labels))
#%%
#Save previous models/pipeline
import dill 

with open('Forest_reg_full_pipe.pkl', 'wb') as file:
    dill.dump(Forest_reg_full_pipe, file)
#%%

'''
Ex 5

Automatically explore some preparation options using GridSearchCV .

'''
                       
                        
param_distribs_prerp = {'preparation_with_selection__preprocessor__numeric__imputer__strategy': ['median','mean','most_frequent'],
        'preparation_with_selection__selector__kw_args':[dict(k=i) for i in range(1,len(feature_importances)+1)],
    }

Ex_5_prep = GridSearchCV(Pipeline_full, param_distribs_prerp, cv=5, scoring='neg_mean_squared_error',verbose=2, n_jobs=4)
Ex_5_prep.fit(housing,housing_labels)
#%%

#%%
print('Best_estimator:',Ex_5_prep.best_estimator_)
print('Best params:', Ex_5_prep.best_params_)
print('Best_score_rmse:',np.sqrt(-Ex_5_prep.best_score_))

cvres = Ex_5_prep.cv_results_
for score, params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-score),params)
#%%