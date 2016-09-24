import pandas as pd
import numpy as np
import metrics
import cuisine_correlation
import os

from sklearn.cross_validation import train_test_split
from sklearn import linear_model, ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor


print(" ### Phase : Loading data.. ###")

base_dir = os.path.dirname(os.path.realpath('__file__'))

id_map_path = os.path.join(base_dir, "restaurant_ids_to_yelp_ids.csv")

id_map = pd.read_csv(id_map_path)
id_dict = {}

# each Yelp ID may correspond to up to 4 Boston IDs
for i, row in id_map.iterrows():
    # get the Boston ID
    boston_id = row["restaurant_id"]

    # get the non-null Yelp IDs
    non_null_mask = ~pd.isnull(row.ix[1:])
    yelp_ids = row[1:][non_null_mask].values

    for yelp_id in yelp_ids:
        id_dict[yelp_id] = boston_id

# Read business file
business_file_path = os.path.join(base_dir, "yelp_boston_academic_dataset/yelp_academic_dataset_business.json")
with open(business_file_path, 'r') as business_file:
    # the file is not actually valid json since each line is an individual
    # dict -- we will add brackets on the very beginning and ending in order
    # to make this an array of dicts and join the array entries with commas
    business_json = '[' + ','.join(business_file.readlines()) + ']'

business = pd.read_json(business_json)

# Opening reviews file
review_file_path = os.path.join(base_dir,"yelp_boston_academic_dataset/yelp_academic_dataset_review.json")
with open(review_file_path, 'r') as review_file:
    # the file is not actually valid json since each line is an individual
    # dict -- we will add brackets on the very beginning and ending in order
    # to make this an array of dicts and join the array entries with commas
    review_json = '[' + ','.join(review_file.readlines()) + ']'

reviews = pd.read_json(review_json)

reviews.drop(['review_id', 'type'],
             inplace=True,
             axis=1)


## Fetching cuisine info

cuisines_info = cuisine_correlation.get_cuisine_info()

# replace yelp business_id with boston restaurant_id
map_to_boston_ids = lambda yelp_id: id_dict[yelp_id] if yelp_id in id_dict else np.nan
reviews.business_id = reviews.business_id.map(map_to_boston_ids)
cuisines_info.business_id = cuisines_info.business_id.map(map_to_boston_ids)
business.business_id = business.business_id.map(map_to_boston_ids)


# rename first column to restaurant_id so we can join with boston data
reviews.columns = ["restaurant_id", "date", "stars", "text","user_id","votes"]

# drop restaurants not found in boston data
reviews = reviews[pd.notnull(reviews.restaurant_id)]

cuisines_info = cuisines_info[pd.notnull(cuisines_info.business_id)]


# # Read AllViolations.csv
all_violation_path = os.path.join(base_dir, "AllViolations.csv")

violations = pd.read_csv(all_violation_path, index_col=0)

def char_count(word):
    return sum(c != ' ' for c in word)

print("### Phase : Feature Building... ###")

avg_stars = list()
number_of_reviews = list()
total_chars = list()
usefulness = list()
review_count_useful = list()
review_count_cool = list()
business_rating = list()
rating_diff = list()

def getValidReviews(violations, reviews):

    N = len(violations)

    for i, (pid, row) in enumerate(violations.iterrows()):
        # we want to only get reviews for this restaurant that occurred before the inspection
        if(i%3000 == 0):
            print(i)
        pre_inspection_mask = (reviews.date < row.date) & (reviews.restaurant_id == row.restaurant_id)

        business_rating.append((business[business.business_id == row.restaurant_id]['stars']).tolist()[0])

        # pre-inspection reviews
        pre_inspection_reviews = reviews[pre_inspection_mask]
        if(len(pre_inspection_reviews) > 0):
            num_reviews = len(pre_inspection_reviews)
            avg_stars.append(np.mean(pre_inspection_reviews['stars']))
            number_of_reviews.append(num_reviews)
            a = list(map(char_count, pre_inspection_reviews['text'].tolist()))
            temp_useful = [d['useful'] for d in pre_inspection_reviews['votes']]
            usefulness.append(sum(temp_useful))
            total_chars.append(sum(a))
            review_count_useful.append((np.sum(temp_useful) + 1)/num_reviews)
            temp_cool = [d['cool'] for d in pre_inspection_reviews['votes']]
            review_count_cool.append((np.sum(temp_cool) + 1)/num_reviews)
        else:
            avg_stars.append(0)
            number_of_reviews.append(0)
            total_chars.append(0)
            usefulness.append(0)
            review_count_useful.append(0)
            review_count_cool.append(0)

getValidReviews(violations, reviews)

violations['avg_stars'] = avg_stars
violations['number_of_reviews'] = number_of_reviews
violations['total_chars'] = total_chars
violations['usefulness'] = usefulness
violations['review_count_useful'] = review_count_useful
violations['review_count_cool'] = review_count_cool
violations['business_rating'] = business_rating
violations['rating_diff'] = np.subtract(business_rating,avg_stars)


violations= violations.merge(cuisines_info, left_on='restaurant_id', right_on='business_id', how='outer')

# # Drop columns no longer needed
violations.drop(['date', 'restaurant_id','business_id'],
             inplace=True,
             axis=1)


# # Splitting data to train and test
print("Splitting into training and testing set")
train, test = train_test_split(violations, test_size = 0.2, random_state=34)


cols = [col for col in train.columns if col not in ['*', '**', '***']]

actual = test.as_matrix(['*','**','***'])
print("Ready to build models..")

# Linear model 1.21 to 1.178 after cuisine addition
print("### Phase: Model Building ###")
print(" LINEAR MODEL")

model_lm = linear_model.LinearRegression()

model_lm.fit(train[cols], train[['*','**','***']])

predictions = model_lm.predict(test[cols])

print("Linear Regression Results : ", metrics.weighted_rmsle(np.around(predictions), actual, metrics.KEEPING_IT_CLEAN_WEIGHTS))



# Decision Tree
print(" DECISION TREE")

regr_1 = DecisionTreeRegressor(random_state=34, splitter='random',max_features='log2', max_leaf_nodes=30000)

regr_1.fit(train[cols], train[['*','**','***']])
predictions = regr_1.predict(test[cols])

print("Decision Tree Results: ", metrics.weighted_rmsle(np.around(predictions), actual, metrics.KEEPING_IT_CLEAN_WEIGHTS))


# PLS Regression
print(" 3. PARTIAL LEAST SQUARES")

pls2 = PLSRegression(max_iter=1000)

pls2.fit(train[cols], train[['*','**','***']])
predictions = pls2.predict(test[cols])

predictions = predictions.clip(min = 0)
print("PLS regression : ", metrics.weighted_rmsle(np.around(predictions), actual, metrics.KEEPING_IT_CLEAN_WEIGHTS))


# Ridge Regression

print("  RIDGE REGRESSION")
clf = Ridge(random_state=34, solver = 'sag')
clf.fit(train[cols], train[['*','**','***']])
predictions = clf.predict(test[cols])

print("Ridge regression : ", metrics.weighted_rmsle(np.around(predictions), actual, metrics.KEEPING_IT_CLEAN_WEIGHTS))

# Lasso Regression

print(" LASSO REGRESSION")
clf = linear_model.Lasso(random_state=34)
clf.fit(train[cols], train[['*','**','***']])
predictions = clf.predict(test[cols])

print("Lasso regression : ", metrics.weighted_rmsle(np.around(predictions), actual, metrics.KEEPING_IT_CLEAN_WEIGHTS))

#  K-NN
print(" K-NN ")

clf = KNeighborsRegressor(weights = 'distance',p=4)

clf.fit(train[cols], train[['*','**','***']])

predictions = clf.predict(test[cols])

print("K-NN: ", metrics.weighted_rmsle(np.around(predictions), actual, metrics.KEEPING_IT_CLEAN_WEIGHTS))


# SVM

print(" 5. SVM ")
#params = {"gamma":[0.9,1.9],"C":[2,5]}
clf = svm.SVR(kernel='poly')
#clf = GridSearchCV(svm1, param_grid=params,cv=2)

clf.fit(train[cols], train['*'])
predictions1 = clf.predict(test[cols])

clf.fit(train[cols], train['**'])
predictions2 = clf.predict(test[cols])

clf.fit(train[cols], train['***'])
predictions3 = clf.predict(test[cols])

print("Making predictions")

predictions1 = predictions1.clip(min = 0)
predictions2 = predictions2.clip(min = 0)
predictions3 = predictions3.clip(min = 0)

dict = {"A":predictions1, "B":predictions2, "C":predictions3}
df = pd.DataFrame(data=dict)
predictions = df.as_matrix()

print("SVR results : ", metrics.weighted_rmsle(np.around(predictions), actual, metrics.KEEPING_IT_CLEAN_WEIGHTS))


svr = LinearSVR()
print("Making predictions")

svr.fit(train[cols], train['*'])
predictions1 = svr.predict(test[cols])

svr.fit(train[cols], train['**'])
predictions2 = svr.predict(test[cols])

svr.fit(train[cols], train['***'])
predictions3 = svr.predict(test[cols])

predictions1 = predictions1.clip(min = 0)
predictions2 = predictions2.clip(min = 0)
predictions3 = predictions3.clip(min = 0)

dict = {"A":predictions1, "B":predictions2, "C":predictions3}
df = pd.DataFrame(data=dict)
predictions = df.as_matrix()

print("Linear SVR results : ", metrics.weighted_rmsle(np.around(predictions), actual, metrics.KEEPING_IT_CLEAN_WEIGHTS))


# Random Forest model
print("Training RF..")

param_dist = {"max_depth": [10,20,50,100,None],
              "max_features": ['auto','sqrt','log2'],
              "min_samples_leaf": [50,60,80],
              "bootstrap": [True, False],
              "n_estimators":[50]}


clf1 = RandomForestRegressor(random_state=34,n_estimators=1000, max_features='log2',max_depth=500)

clf = GridSearchCV(clf1, param_grid=param_dist)

clf.fit(train[cols], train[['*','**','***']])
print("Making predictions")

predictions = clf.predict(test[cols])

print("RF : ", metrics.weighted_rmsle(np.around(predictions), actual, metrics.KEEPING_IT_CLEAN_WEIGHTS))

#GBM model
param_grid={'n_estimators': [5,10],
            'max_depth':[5,1000],
            'learning_rate':[0.3,0.9],
            'min_samples_leaf':[10,50],
            'min_samples_split':[2,10],
            'max_features':'sqrt',
            'subsample':0.8
        }

clf = ensemble.GradientBoostingRegressor(n_estimators= 1500,max_depth=50,random_state=34,max_features='sqrt',learning_rate=0.3,min_samples_split=2)

clf = GridSearchCV(clf, param_grid =param_grid, scoring = 'mean_squared_error',cv=5)

clf.fit(train[cols], train['*'])
print("Making predictions for 1")
predictions1 = clf.predict(test[cols])

clf.fit(train[cols], train['**'])
print("Making predictions for 2")
predictions2 = clf.predict(test[cols])
#
clf.fit(train[cols], train['***'])
print("Making predictions for 3")
predictions3 = clf.predict(test[cols])

#Removing negative values
predictions1 = predictions1.clip(min = 0)
predictions2 = predictions2.clip(min = 0)
predictions3 = predictions3.clip(min = 0)


dict = {"A":predictions1, "B":predictions2, "C":predictions3}
df = pd.DataFrame(data=dict)
predictions = df.as_matrix()
print("Making final predictions")

print("BEST : ", clf.best_params_)
#print("IMP "+clf.feature_importances_)
print("ESTIMATOR :"+str(clf.best_estimator_))
print("SCORE "+str(clf.best_score_))

print("GBM regression : ", metrics.weighted_rmsle(np.around(predictions), actual, metrics.KEEPING_IT_CLEAN_WEIGHTS))


# STOCHASTIC GRADIENT DESCENT
clf = linear_model.SGDRegressor(random_state=34)
clf.fit(train[cols], train['*'])
print("Making predictions for 1")
predictions1 = clf.predict(test[cols])

clf.fit(train[cols], train['**'])
print("Making predictions for 2")
predictions2 = clf.predict(test[cols])
#
clf.fit(train[cols], train['***'])
print("Making predictions for 3")
predictions3 = clf.predict(test[cols])
#
# #Removing negative values
predictions1 = predictions1.clip(min = 0)
predictions2 = predictions2.clip(min = 0)
predictions3 = predictions3.clip(min = 0)


dict = {"A":predictions1, "B":predictions2, "C":predictions3}
df = pd.DataFrame(data=dict)
predictions = df.as_matrix()
print("Making final predictions")

print("BEST : ", clf.best_params_)
#print("IMP "+clf.feature_importances_)
print("ESTIMATOR :"+str(clf.best_estimator_))
print("SCORE "+str(clf.best_score_))

print("SGD : ", metrics.weighted_rmsle(np.around(predictions), actual, metrics.KEEPING_IT_CLEAN_WEIGHTS))

# ADABOOST

regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=None,random_state=34,max_features='log2', min_samples_split=1),
                          n_estimators=47, random_state=34)



regr_2.fit(train[cols], train['*'])
print("Making predictions for 1")
predictions1 = regr_2.predict(test[cols])

regr_2.fit(train[cols], train['**'])
print("Making predictions for 2")
predictions2 = regr_2.predict(test[cols])
#
regr_2.fit(train[cols], train['***'])
print("Making predictions for 3")
predictions3 = regr_2.predict(test[cols])


dict = {"A":predictions1, "B":predictions2, "C":predictions3}
df = pd.DataFrame(data=dict)
predictions = df.as_matrix()

print("AdaBoost using Decision Tree: ", metrics.weighted_rmsle(np.around(predictions), actual, metrics.KEEPING_IT_CLEAN_WEIGHTS))