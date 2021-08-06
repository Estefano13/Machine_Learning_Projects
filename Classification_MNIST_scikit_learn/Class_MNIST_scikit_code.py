#Classification on MNIST

#%%
#Getting the dataset
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')

#%%

#Saving the dataset 
import dill 

with open('MNIST_dataset.pkl', 'wb') as file:
    dill.dump(mnist, file)
#%%
#loading it
import dill

with open('MNIST_dataset.pkl', 'rb') as file:
    mnist = dill.load(file)
    
#%%

X, y = mnist['data'], mnist['target']
print(X.shape, y.shape)

#%%

#Chacking one of the images
import matplotlib.pyplot as plt

Instance = X[401].reshape(28,28)

plt.imshow(Instance,cmap = 'Greys',interpolation='nearest') 
plt.axis('off')

print(y[401])

#%%
'''
Dividing into training and test set. 

You should always create a test set and set it aside before inspecting the data
closely. The MNIST dataset is actually already split into a training set (the first 60,000
images) and a test set (the last 10,000 images)

You should use fetch_openml() instead. However, it returns the unsorted MNIST dataset, 
whereas fetch_mldata() returned the dataset sorted by target (the training set and the 
test set were sorted separately).
'''
import numpy as np

#turn y into integers from strings
y = y.astype(int)

#Adding an index as it is 
reorder_train_step_1 = [(target, i) for i, target in enumerate(y[:60000])]
#sort it based on label
def target(elem):
    return elem[0]

reorder_train_step_2 = sorted(reorder_train_step_1,key=target)
#Turn into array and extract the index
reorder_train_step_3 = np.array(reorder_train_step_2)[:,1]   

#Adding an index as it is 
reorder_test_step_1 = [(target, i) for i, target in enumerate(y[60000:])]
#sort it based on label
reorder_test_step_2 = sorted(reorder_test_step_1,key=target)
#Turn into array and extract the index
reorder_test_step_3 = np.array(reorder_test_step_2)[:,1]   

#rearrange the set

X_train, X_test, y_train, y_test = X[reorder_train_step_3], X[reorder_test_step_3+60000],y[reorder_train_step_3], y[reorder_test_step_3+60000]
#%%
'''
Let’s also shuffle the training set; this will guarantee that all cross-validation folds will
be similar (you don’t want one fold to be missing some digits). Moreover, some learn‐
ing algorithms are sensitive to the order of the training instances, and they perform
poorly if they get many similar instances in a row. Shuffling the dataset ensures that
this won’t happen.
'''
import numpy as np

new_index = np.random.permutation(60000)
X_train, y_train = X_train[new_index], y_train[new_index]
#%%
##############################My own bullshit#######################################

#Chacking how balanced the cats are
labels, counts = np.unique(y_train,return_counts = True)
balanced_cats = dict(zip(labels, counts))
print(balanced_cats)
#%%

from sklearn.svm import LinearSVC

first_class = LinearSVC().fit(X_train,y_train)

#%%

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score

first_class_predictions = first_class.predict(X_train)
first_class_accuracy = accuracy_score(y_train, first_class_predictions)
first_class_balanced_accuracy = balanced_accuracy_score(y_train, first_class_predictions)
first_class_f1 = f1_score(y_train, first_class_predictions, average = 'macro')


print('Accuracy:', first_class_accuracy,'\nBalanced accuracy:', first_class_balanced_accuracy,'\nF1 Score:', first_class_f1)

#%%
##############################Training a binary classifier##################################

#Lets try to identify number 4

y_train_4 = (y_train == 4)
y_test_4 = (y_test == 4)

#%%

# Training with SGDclassifier

'''
This classifier has the advantage of being capable of handling very large datasets efficiently.
This is in part because SGD deals with training instances independently, one at a time
(which also makes SGD well suited for online learning), as we will see later.
'''

from sklearn.linear_model import SGDClassifier

SDG_class = SGDClassifier(random_state = 42)
SDG_class.fit(X_train,y_train_4)

#%%

# predict 

from numpy.random import randint

some_dig = X_train[randint(0,60001)].reshape(1,784)

prediction_on_one = SDG_class.predict(some_dig)
print(prediction_on_one)

#%%
###########################################Performance metrics##################################

#Essentially this is the cross_val_score function with accuracy as a score

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_4):
    clone_clf = clone(SDG_class)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_4[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_4[test_index])
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)

    print(n_correct / len(y_pred))

#array([0.9785 , 0.97625, 0.97165])

#%%

from sklearn.model_selection import cross_val_score

scores_from_cvs = cross_val_score(SDG_class, X_train, y_train_4, cv=3, scoring="accuracy")
print(scores_from_cvs)
#array([0.9785 , 0.97625, 0.97165])

#%%

# Accuracy is bullshit

from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


scores_from_cvs_exp = cross_val_score(Never5Classifier(), X_train, y_train_4, cv=3, scoring="accuracy")
print(scores_from_cvs_exp)
#[0.90375 0.8995  0.90465] 

'''
That’s right, it has over 90% accuracy! This is simply because only about 10% of the
images are 5s, so if you always guess that an image is not a 5, you will be right about
90% of the time. Beats Nostradamus.
This demonstrates why accuracy is generally not the preferred performance measure
for classifiers, especially when you are dealing with skewed datasets (i.e., when some
classes are much more frequent than others).
'''
#%%
####################################Confusion matrix##################################

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

cv_spec = StratifiedKFold(n_splits=3, random_state=42, shuffle=(True))


y_train_pred = cross_val_predict(SDG_class, X_train, y_train_4, cv=3)
print(confusion_matrix(y_train_4, y_train_pred))

'''
The function cross_val_predict has a similar interface to cross_val_score, but returns, 
for each element in the input, the prediction that was obtained for that element when it 
was in the test set. Only cross-validation strategies that assign all elements to a test 
set exactly once can be used (otherwise, an exception is raised).
'''
#%%

##############################Precision########################################

'''
The confusion matrix gives you a lot of information, but sometimes you may prefer a
more concise metric. An interesting one to look at is the accuracy of the positive pre‐
dictions; this is called the precision of the classifier.

precision = TP / (TP + FP)

Recall or sensitivity is

recall = TP / (TP + FN)
'''

from sklearn.metrics import precision_score, recall_score

prec_score_kfold = precision_score(y_train_4, y_train_pred)
recall_score_kfold = recall_score(y_train_4, y_train_pred)
print(prec_score_kfold, recall_score_kfold)
#0.8977624158631982 0.844744950359466
#%%

'''
It is often convenient to combine precision and recall into a single metric called the F 1
score, in particular if you need a simple way to compare two classifiers. The F 1 score is
the harmonic mean of precision and recall. Whereas the regular mean
treats all values equally, the harmonic mean gives much more weight to low values.
As a result, the classifier will only get a high F 1 score if both recall and precision are
high.
'''
from sklearn.metrics import f1_score

f1_score_kfold = f1_score(y_train_4, y_train_pred)
print(f1_score_kfold)
#%%
#############################DEcision boundary and the precision/recall trade off##################

'''
Scikit-Learn does not let you set the threshold directly, but it does give you access to
the decision scores that it uses to make predictions. Instead of calling the classifier’s
predict() method, you can call its decision_function() method, which returns a
score for each instance, and then make predictions based on those scores using any
threshold you want. The SGDClassifier uses a threshold equal to 0.
'''

y_scores = SDG_class.decision_function(some_dig)
print(y_scores)

threshold = 0
y_scores_pred = (y_scores > threshold)
print(y_scores_pred)

# A diff one

threshold = -20000
y_scores_pred = (y_scores > threshold)
print(y_scores_pred)

#%%
#To change the decision boundary on the entire set of preditions

y_scores_pred_thresh = cross_val_predict(SDG_class, X_train, y_train_4, cv=3,method="decision_function")

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholdss = precision_recall_curve(y_train_4, y_scores_pred_thresh)
#%%

plt.plot(thresholdss,recalls[:-1],'b:',label="Recalls")
plt.plot(thresholdss,precisions[:-1],'g-',label = "Precision")
plt.xlabel('Thresholds')
plt.legend(loc="upper left")
plt.ylim([-0.1, 1.1])
plt.xlim([-90000, 90000])

#%%

'''
Now you can simply select the threshold value that gives you the best precision/recall
tradeoff for your task. Another way to select a good precision/recall tradeoff is to plot
precision directly against recall.
'''
plt.plot(recalls,precisions,'b')
plt.xlabel('recalls')
plt.ylabel('precisions')
#%%
#Say I want 90% precision

index_o_int = np.where(precisions > 0.9)
threshold_oi = thresholdss[index_o_int[0][0]]

y_train_90 = (y_scores_pred_thresh > threshold_oi)

prec_score_90 = precision_score(y_train_4, y_train_90)
recall_score_90 = recall_score(y_train_4, y_train_90)
print(prec_score_90, recall_score_90)

#We can make a clssifier at any precission now

#%%
##################################ROC curve####################################

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_4, y_scores_pred_thresh)

plt.plot(fpr, tpr, linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

#%%
from sklearn.metrics import roc_auc_score

print(roc_auc_score(y_train_4, y_scores_pred_thresh))
#0.9886273832359777

'''
This is a very good value.

Since the ROC curve is so similar to the precision/recall (or PR)
curve, you may wonder how to decide which one to use. As a rule
of thumb, you should prefer the PR curve whenever the positive
class is rare or when you care more about the false positives than
the false negatives, and the ROC curve otherwise.
'''
#%%
#Let's try a random forest

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_4, cv=3,method="predict_proba")
#%%

#Here the probabilities of the positive class, menaning the 4 are stored in the second column

y_score_forest = y_probas_forest[:,1]
fpr_for, tpr_for, thresholds_for = roc_curve(y_train_4, y_score_forest)

plt.plot(fpr_for, tpr_for,'g:', linewidth=2,label='Random Forest')
plt.plot(fpr, tpr,'b-', linewidth=2,label='SGD')
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

print(roc_auc_score(y_train_4, y_score_forest))
#0.9984420244447128
#%%
y_scoress_forest = cross_val_predict(forest_clf, X_train, y_train_4, cv=3)

prec_score_for = precision_score(y_train_4, y_scoress_forest)
recall_score_for = recall_score(y_train_4, y_scoress_forest)
print(prec_score_for, recall_score_for)
#%%
###############################Multiclass classification###############################

SDG_class.fit(X_train,y_train)

#%%
print(SDG_class.predict(some_dig))

#It is a One v Rest strat

print(SDG_class.decision_function(some_dig),'\n',SDG_class.classes_)

#%%
#Forcing a OVO

from sklearn.multiclass import OneVsOneClassifier

ovo_class = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_class.fit(X_train,y_train) 
print(ovo_class.predict(some_dig))
print(len(ovo_class.estimators_))
#%%

#Random  which naturally handles multiclass

from sklearn.ensemble import RandomForestClassifier

forest_classs = RandomForestClassifier(random_state=42)
forest_classs.fit(X_train,y_train)
print(forest_classs.predict(some_dig))
print(forest_classs.predict_proba(some_dig))

#%%

print(cross_val_score(SDG_class, X_train, y_train, cv=3, scoring="accuracy"))
#[0.87335 0.84235 0.8553 ] Not supper good

#Trying to improve
#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data_X = scaler.fit_transform(X_train)
print(cross_val_score(SDG_class, scaled_data_X, y_train, cv=3, scoring="accuracy"))

#[0.8962  0.90685 0.90265] Not supper good but better than before
#%%
########################Error analysis#####################################

#Confusion matrix

y_train_pred_all = cross_val_predict(SDG_class, X_train, y_train, cv=3)
print(confusion_matrix(y_train, y_train_pred_all))
plt.matshow(confusion_matrix(y_train, y_train_pred_all), cmap=plt.cm.gray)
plt.show()
#%%
SDG_class.fit(X_train,y_train)
y_train_pred_all_no_cv = SDG_class.predict(X_train)
print(confusion_matrix(y_train, y_train_pred_all_no_cv))

plt.matshow(confusion_matrix(y_train, y_train_pred_all_no_cv), cmap=plt.cm.gray)
plt.show()
#%%
#Getting rates of error per cat
conf_mat = confusion_matrix(y_train, y_train_pred_all)

row_sum = conf_mat.sum(axis=1, keepdims = True)
norm_conf_mat = conf_mat/row_sum

np.fill_diagonal(norm_conf_mat,0)
plt.matshow(norm_conf_mat, cmap=plt.cm.gray)
plt.xlabel('Predicted classes')
plt.ylabel('Actual classes')
plt.show()

'''
Analyzing the confusion matrix can often give you insights on ways to improve your
classifier. Looking at this plot, it seems that your efforts should be spent on improving
classification of 8s and 9s, as well as fixing the specific 3/5 confusion. For example,
you could try to gather more training data for these digits. Or you could engineer
new features that would help the classifier—for example, writing an algorithm to
count the number of closed loops (e.g., 8 has two, 6 has one, 5 has none). Or you
could preprocess the images (e.g., using Scikit-Image, Pillow, or OpenCV) to make
some patterns stand out more, such as closed loops.
'''
#%%
########################Working on the 3 to 5 errors
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    #What does this 2 do
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = 'Greys', **options)
    plt.axis("off")
    
cl_a, cl_b = 3, 5
X_aa = X_train[(y_train==cl_a) & (y_train_pred_all == cl_a)]
X_ab = X_train[(y_train==cl_a) & (y_train_pred_all == cl_b)]
X_ba = X_train[(y_train==cl_b) & (y_train_pred_all == cl_a)]
X_bb = X_train[(y_train==cl_b) & (y_train_pred_all == cl_b)]

plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)

plt.show()

'''
The main difference between 3s and 5s is the position of the small line that joins the
top line to the bottom arc. If you draw a 3 with the junction slightly shifted to the left,
the classifier might classify it as a 5, and vice versa. In other words, this classifier is
quite sensitive to image shifting and rotation. So one way to reduce the 3/5 confusion
would be to preprocess the images to ensure that they are well centered and not too
rotated. This will probably help reduce other errors as well.
'''

#%%
##############Multilabel calssification##########################

from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
#%%

print(knn_clf.predict(some_dig))

#%%

y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)
print(f1_score(y_train, y_train_knn_pred, average="macro"))

#%%
##################Multioutput classification#####################

'''
To illustrate this, let’s build a system that removes noise from images. It will take as
input a noisy digit image, and it will (hopefully) output a clean digit image, repre‐
sented as an array of pixel intensities, just like the MNIST images. Notice that the
classifier’s output is multilabel (one label per pixel) and each label can have multiple
values (pixel intensity ranges from 0 to 255). It is thus an example of a multioutput
classification system.
'''

import numpy.random as rnd

noise_Train = rnd.randint(0, 100, (len(X_train), 784))
noise_test = rnd.randint(0, 100, (len(X_test), 784))
X_train_mod = X_train + noise_Train
X_test_mod = X_test + noise_test
y_train_mod = X_train
y_test_mod = X_test

#%%

import matplotlib.pyplot as plt

Instance_mod = X_train_mod[401].reshape(28,28)
plt.figure(figsize=(4,8))
plt.subplot(221);plt.imshow(Instance_mod,cmap = 'Greys',interpolation='nearest') 
plt.axis('off')

Instance = X_train[401].reshape(28,28)
plt.subplot(222);plt.imshow(Instance,cmap = 'Greys',interpolation='nearest') 
plt.axis('off')

#%%

from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)
test_num = X_test_mod[rnd.randint(0,10001)]
clean_digit = knn_clf.predict([test_num])
#%%
plt.imshow(test_num.reshape(28,28),cmap = 'Greys',interpolation='nearest') 
#%%
plt.imshow(clean_digit.reshape(28,28),cmap = 'Greys',interpolation='nearest') 

#%%
#######Excercises######

#1
'''
Try to build a classifier for the MNIST dataset that achieves over 97% accuracy
on the test set. Hint: the KNeighborsClassifier works quite well for this task;
you just need to find good hyperparameter values (try a grid search on the
weights and n_neighbors hyperparameters).
'''

#Tunning hyperparameters with GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

parameter_grid = [
    {'n_neighbors':[3,4,5],'weights':['uniform','distance']}]

#KNN_reg_grid = GridSearchCV(KNeighborsClassifier(), parameter_grid, cv = 5, scoring='neg_mean_squared_error',refit=True)
KNN_reg_grid = GridSearchCV(KNeighborsClassifier(), parameter_grid, cv = 5, verbose=3, n_jobs=-1, scoring='accuracy',refit=True)
KNN_reg_grid.fit(X_train,y_train)
#%%
'''
I dont want to wait for the gridsearch to complete so Ill just input the best params and save those
'''
from sklearn.neighbors import KNeighborsClassifier

KNN_reg = KNeighborsClassifier(n_neighbors=4, weights='distance')
KNN_reg.fit(X_train,y_train)
#%%
import dill 

with open('KNN_reg_grid_mod.pkl', 'wb') as file:
    dill.dump(KNN_reg, file)
#%%
import dill

with open('KNN_reg_grid_mod.pkl', 'rb') as file:
    KNN_reg_grid = dill.load(file)

#%%


print('Best_estimator:',KNN_reg_grid.best_estimator_)
print('Best params:', KNN_reg_grid.best_params_)
print('Best_score_rmse:',np.sqrt(-KNN_reg_grid.best_score_))

cvres = KNN_reg_grid.cv_results_
for score, params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-score),params)
    
#%%

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score

KNN_reg_grid_pred = KNN_reg.predict(X_test)
KNN_reg_grid_accuracy = accuracy_score(y_test, KNN_reg_grid_pred)
KNN_reg_grid_balanced_accuracy = balanced_accuracy_score(y_test, KNN_reg_grid_pred)
KNN_reg_grid_f1 = f1_score(y_test, KNN_reg_grid_pred, average = 'macro')


print('Accuracy:', KNN_reg_grid_accuracy,'\nBalanced accuracy:', KNN_reg_grid_balanced_accuracy,'\nF1 Score:', KNN_reg_grid_f1)
'''
Accuracy: 0.9714 
Balanced accuracy: 0.9710823052664403 
F1 Score: 0.971224084176584
'''

#%%
#2
'''
Write a function that can shift an MNIST image in any direction (left, right, up,
or down) by one pixel. (*) Then, for each image in the training set, create four shif‐
ted copies (one per direction) and add them to the training set. Finally, train your
best model on this expanded training set and measure its accuracy on the test set.
You should observe that your model performs even better now! This technique of
artificially growing the training set is called data augmentation or training set
expansion.

(*)
You can use the shift() function from the scipy.ndimage.interpolation module. For example,
shift(image, [2, 1], cval=0) shifts the image 2 pixels down and 1 pixel to the right.
'''