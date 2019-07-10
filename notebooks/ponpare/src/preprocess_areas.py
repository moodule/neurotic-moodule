from preprocessing_lib import *


__author__ = "EelMood"


# COMPUTE AREA MAPPINGS  ======================================================


def prefectures_to_gps(prefectures_df):
    latitude_dict = dict()
    longitude_dict = dict()
    pref_key = prefectures_df.keys()[0]     # there's a mismatch between "PREF_NAME' and the first key = extra space...

    print('[setup] \t creating a dictionary to map "PREF_NAME" values to GPS')

    for row in prefectures_df.iterrows():
        pref_name = row[1][pref_key]
        latitude_dict[pref_name] = row[1]['LATITUDE']
        longitude_dict[pref_name] = row[1]['LONGITUDE']

    return latitude_dict, longitude_dict


def areas_to_gps(coupon_area_df, latitude_dict, longitude_dict):
    coupons_area_dict = dict()

    print('[setup] \t creating a dictionary to map "COUPON_ID_hash" values to [GPS tuples] wrt coupon_areas')

    coupon_area_df['LATITUDE'] = coupon_area_df['PREF_NAME'].map(latitude_dict)
    coupon_area_df['LONGITUDE'] = coupon_area_df['PREF_NAME'].map(longitude_dict)

    for index, row in coupon_area_df[['COUPON_ID_hash', 'LATITUDE', 'LONGITUDE']].iterrows():

        # get current coupon's properties
        coupon_hash = row['COUPON_ID_hash']
        coupon_lat = row['LATITUDE']
        coupon_long = row['LONGITUDE']

        # add the area's coordinates to the coupon's list
        temp = coupons_area_dict.get(coupon_hash, [])
        temp.append((coupon_lat, coupon_long))

        # update the list for the current coupon
        coupons_area_dict[coupon_hash] = temp

    return coupons_area_dict


def coupons_to_gps(coupons_df, latitude_dict, longitude_dict):
    coupons_area_dict = dict()

    print('[setup] \t creating a dictionary to map "COUPON_ID_hash" values to [GPS tuples] wrt ken_name')

    for index, row in coupons_df[['COUPON_ID_hash', 'ken_name']].iterrows():

        # get current coupon's properties
        coupon_hash = row['COUPON_ID_hash']
        ken_name = row['ken_name']
        coupon_lat = latitude_dict[ken_name]
        coupon_long = longitude_dict[ken_name]

        coupons_area_dict[coupon_hash] = [(coupon_lat, coupon_long)]

    return coupons_area_dict


# COMPUTE MEAN DISTANCE USER / COUPON =========================================


# mean_distance = 0.
# count = 1
#
# for coupon_index, coupon_id in coupons_train['COUPON_ID_hash'].iteritems():
#     for user_index, user_id in users['USER_ID_hash'].iteritems():
#         user_lat = users['LATITUDE'].get_value(user_index)
#         user_long = users['LONGITUDE'].get_value(user_index)
#         distance = min_distance((user_lat, user_long), coupons_area_train_dict.get(coupon_id, []))
#         if distance >= 0.:
#             mean_distance += distance
#             count += 1
#             if count % 10000 == 0:
#                 print('iteration {} : mean_distance = {}'.format(count, (mean_distance / count)) + '...')
#
# if count > 0:
#     mean_distance = mean_distance / count
