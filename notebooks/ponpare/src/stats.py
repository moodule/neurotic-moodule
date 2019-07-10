__author__ = 'EelMood'

import os, sys
import pandas as pd
import numpy as np
import numpy.random as rng
import math
import matplotlib.pyplot as plt
from preprocessing_functions import *


# TODO calculer le nombre moyen d'utilisateur faisant des achats, par semaine


print('=========================================================== LOADING DATA')


print('loading users\' data...')
users = pd.read_csv('../input/translated/user_list_translated.csv')

print('loading training coupons\' data...')
coupons_train = pd.read_csv('../input/translated/coupon_list_train_translated.csv')

print('loading testing coupons\' data...')
coupons_test = pd.read_csv('../input/translated/coupon_list_test_translated.csv')

print('loading user visits\' data...')
visits = pd.read_csv('../input/coupon_visit_train.csv')

print('loading user transactions\' data...')
transactions = pd.read_csv('../input/translated/coupon_detail_train_translated.csv')

print('loading prefecture locations...')
prefectures = pd.read_csv('../input/translated/prefecture_locations_translated.csv')

print('loading training coupon locations\' data...')
coupons_area_train = pd.read_csv('../input/translated/coupon_area_train_translated.csv')

print('loading testing coupon locations\' data...')
coupons_area_test = pd.read_csv('../input/translated/coupon_area_test_translated.csv')

print('done.')


print('=========================================================== SOME GLOBALS')


# first day of the dataset
start_timestamp = 1309478400.

# coupons and users id sets
users_id_set = set(users['USER_ID_hash'].unique())
coupons_id_set = set(coupons_train['COUPON_ID_hash'].unique())


print('========================================================== PREPROCESSING')


# convert the purchase date to timestamp
columns_to_timestamp(transactions, columns=['I_DATE'], origin='datetime')

# there's a mismatch between 'PREF_NAME' and the first key of prefectures
print('getting the actual column name for "PREF_NAME"...')
pref_key = prefectures.keys()[0]

# latitude and longitude dictionaries for coupons
print('creating dictionaries to translate "PREF_NAME" to its gps coordinates...')
latitude_dict = dict()
longitude_dict = dict()
for pref_index, pref_name in prefectures[pref_key].iteritems():
    latitude_dict[pref_name] = prefectures['LATITUDE'].get_value(pref_index)
    longitude_dict[pref_name] = prefectures['LONGITUDE'].get_value(pref_index)

# map the "PREF_NAME" to gps coorinates
print('mapping "PREF_NAME" to "LATITUDE" and "LONGITUDE"...')
coupons_area_train['LATITUDE'] = coupons_area_train['PREF_NAME'].map(latitude_dict)
coupons_area_test['LATITUDE'] = coupons_area_test['PREF_NAME'].map(latitude_dict)
coupons_area_train['LONGITUDE'] = coupons_area_train['PREF_NAME'].map(longitude_dict)
coupons_area_test['LONGITUDE'] = coupons_area_test['PREF_NAME'].map(longitude_dict)


# create a dictionary of the areas of each coupon
coupons_area_dict = dict()
print('creating a dictionary COUPON_ID : [(LATITUDE, LONGITUDE)] for the coupons...')
print('\tin the training set...')
for coupon_index, coupon_id in coupons_area_train['COUPON_ID_hash'].iteritems():
    temp = coupons_area_dict.get(coupon_id, [])
    x = coupons_area_train['LATITUDE'].get_value(coupon_index)
    y = coupons_area_train['LONGITUDE'].get_value(coupon_index)
    temp += [(x, y)]
    coupons_area_dict[coupon_id] = temp
print('\tin the testing set...')
for coupon_index, coupon_id in coupons_area_test['COUPON_ID_hash'].iteritems():
    temp = coupons_area_dict.get(coupon_id, [])
    x = coupons_area_test['LATITUDE'].get_value(coupon_index)
    y = coupons_area_test['LONGITUDE'].get_value(coupon_index)
    temp += [(x, y)]
    coupons_area_dict[coupon_id] = temp


print('done.')


print('========================================================= PURCHASE STATS')

# initialize buyers per week stats
buyers_per_week = dict()
for i in range(52):
    buyers_per_week[i] = []

# get the user ids of the buyers, for each week
for row in transactions.iterrows():
    week_id = week_number(row[1]['I_DATE'], start_timestamp)
    buyers_per_week[week_id].append(row[1]['USER_ID_hash'])

# mean number of purchases per user and per week
purchases_per_user_per_week = dict()
for week_id, buyers_list in buyers_per_week.iteritems():
    number_purchases = len(buyers_list)
    number_users = len(set(buyers_list))
    purchases_per_user_per_week[week_id] = float(number_purchases) / float(number_users)
mean_purchases_per_user_per_week = sum(purchases_per_user_per_week.values()) / len(purchases_per_user_per_week)
stdv_purchases_per_user_per_week = 0.
for week_id, week_purchases in purchases_per_user_per_week.iteritems():
    stdv_purchases_per_user_per_week = stdv_purchases_per_user_per_week + (week_purchases - mean_purchases_per_user_per_week) ** 2
stdv_purchases_per_user_per_week = stdv_purchases_per_user_per_week / 52.
stdv_purchases_per_user_per_week = math.sqrt(stdv_purchases_per_user_per_week)

print('done.')


print('===================================================== PURCHASE CENTROIDS')


# dictionary to hold the bought coupons by each user
coupons_centroids = dict()
for coupon_id in list(coupons_id_set):
    coupons_centroids[coupon_id] = []
users_centroids = dict()
users_purchases = dict()
for user_id in list(users_id_set):
    users_centroids[user_id] = []
    users_purchases[user_id] = []

print('merging users and transactions...')
x_transactions = pd.merge(transactions, users, how='inner', on=['USER_ID_hash'])

print('merging users and transactions...')
x_transactions = pd.merge(x_transactions, coupons_train, how='inner', on=['COUPON_ID_hash'])

# centroids of the areas of availability for each coupon
for coupon_id in list(coupons_id_set):
    if coupon_id in coupons_area_dict:
        coord_list = coupons_area_dict[coupon_id]
        coupons_centroids[coupon_id] = list(centroid(coord_list))

# list of the centroids of the coupons bought by each user
for row in x_transactions.iterrows():
    user_id = row[1]['USER_ID_hash']
    coupon_id = row[1]['COUPON_ID_hash']
    if coupon_id in coupons_centroids:
        temp = users_purchases[user_id]
        coupon_centroid = coupons_centroids[coupon_id]
        if coupon_centroid:
            temp.append(coupon_centroid)
        users_purchases[user_id] = temp

for user_id, coord_list in users_purchases.iteritems():
    print (user_id)
    users_centroids[user_id] = list(centroid(coord_list))

print('done.')


print('=================================================================== PLOT')

# scatter plots
coupon_id = rng.choice(np.array(list(coupons_id_set)), 1)[0]
user_id = rng.choice(np.array(list(users_id_set)), 1)[0]
coupon_x = []
coupon_y = []
user_x = []
user_y = []
for coord in users_purchases[user_id]:
    user_x.append(coord[0])
    user_y.append(coord[1])
for coord in coupons_area_dict[coupon_id]:
    coupon_x.append(coord[0])
    coupon_y.append(coord[1])

# plot parameters
fig, ax = plt.subplots()
bar_width = 0.35
opacity = 0.4

# plot data
index = []
values = []

# number of unique buyers per week
index = np.arange(len(buyers_per_week))
for i in range(len(buyers_per_week)):
    values.append(len(set(buyers_per_week[i])))

# setup plot
rects1 = plt.bar(index, values, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Unique buyers')

plt.xlabel('Week number')
plt.ylabel('Number of unique buyers')
plt.title('Number of buyers per week')
plt.legend()

plt.tight_layout()
plt.show()