__author__ = 'EelMood'

import os, sys
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib as plot

users           = pd.read_csv('../user_list_translated.csv')
coupons_train   = pd.read_csv('../coupon_list_train_translated.csv')
coupons_test    = pd.read_csv('../coupon_list_test_translated.csv')
visits          = pd.read_csv('../input/coupon_visit_train.csv')
transactions    = pd.read_csv('../coupon_detail_train_translated.csv')
coupons_area_train  = pd.read_csv('../coupon_area_train_translated.csv')
coupons_area_test   = pd.read_csv('../coupon_area_test_translated.csv')

# TIME ======================================================================

dates = [dt.datetime.strptime(date,'%Y-%m-%d %H:%M:%S') for date in list(transactions.I_DATE.unique())]
timestamps = [((date - dt.datetime(1970, 1, 1))) for date in dates]
timestamps = [delta.total_seconds() for delta in timestamps]
time_span = dt.datetime.fromtimestamp(max(timestamps)) - dt.datetime.fromtimestamp(min(timestamps))

# USERS ===============================================================

# all users
users_id_set = set(users.USER_ID_hash.unique())
users_count = len(users_id_set)
# users that have visited a page
active_users_id_set = users_id_set.intersection(set(visits.USER_ID_hash.unique()))
inactive_users_id_set = users_id_set.difference(set(visits.USER_ID_hash.unique()))
active_users_count = len(active_users_id_set)
inactive_users_count = len(inactive_users_id_set)
# users that have bought a coupon
buying_users_id_set = set(transactions.USER_ID_hash.unique())
buying_users_count = len(buying_users_id_set)

# PURCHASES ===============================================================

# all transactions
transactions_id_set = set(transactions.PURCHASEID_hash.unique())
transactions_count = len(transactions_id_set)
# mean transactions count
mean_transactions_per_month =  float(transactions_count) / 12.
mean_transactions_per_week = float(transactions_count) / 51.
mean_transactions_per_day = float(transactions_count) / float(time_span.days)
# transactions counts per month to estimate the correlation
transactions_per_month = dict()
transactions_dates = [dt.datetime.strptime(date,'%Y-%m-%d %H:%M:%S') for (k,date) in transactions.I_DATE.iteritems()]
for i in xrange(12):
    transactions_per_month[i+1] = 0
for date in transactions_dates:
    transactions_per_month[date.month] += 1
# transactions per genre per month
transactions_per_genre = dict()
transactions.COUPON_ID_hash.map(coupons_train.COUPON_ID_hash)

# COUPONS ===============================================================

# PLOT ===============================================================

# OLD ===============================================================

# mask = visits['PURCHASE_FLG'].map(lambda x: x==1)
# buying_users_id_set = set(visits[mask]['USER_ID_hash'])

print (visits['USER_ID_hash'].unique().shape)