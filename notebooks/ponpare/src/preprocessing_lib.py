import os, sys
import datetime as dt
import math
import numpy as np
import pandas as pd


__author__ = 'EelMood'


# todo changer les iterations en iterrows, pour etre sur de leur action


# CONSTANTS ===================================================================


earth_radius = 6371.


# FEEDBACK ====================================================================


def display_progress(percent):
    bars_nb = int(math.floor(percent / 5.))
    progress_str = '\r[{0: <20}] {1}%'.format('#'*bars_nb, math.floor(percent))
    sys.stdout.write(progress_str)
    sys.stdout.flush()


# DATE CONVERSION =============================================================


def date_to_timestamp(date_str):
    date = dt.datetime.strptime(date_str, '%Y-%m-%d')
    timestamp = float((date - dt.datetime(1970, 1, 1)).total_seconds())
    return timestamp


def datetime_to_timestamp(date_str):
    date = dt.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    timestamp = float((date - dt.datetime(1970, 1, 1)).total_seconds())
    return timestamp


def columns_to_timestamp(df, columns, origin='date'):
    if df is not None:
        for col in columns:
            print('[mapping] \t converting {} to timestamp...'.format(col))
            if origin == 'date':
                df[col] = df[col].map(lambda x: x if type(x) == float else date_to_timestamp(x))
            elif origin == 'datetime':
                df[col] = df[col].map(lambda x: x if type(x) == float else datetime_to_timestamp(x))


def week_number(timestamp, ref_timestamp):
    week_id = timestamp - ref_timestamp     # time span in seconds between the two timestamps
    week_id = week_id / (7. * 24. * 3600.)  # time span in fractions of weeks (can be negative)
    week_id = math.floor(week_id)           # id of the week corresponding to the argument timespan, wrt to ref timespan

    return week_id


# AREAS PROCESSING =========================================================


def dist_from_coordinates(lat1, lon1, lat2, lon2):
    # if one coordinate is unknown, it returns -1
    haversine = -1.

    if (lat1 != -1) and (lon1 != -1) and (lat2 != -1) and (lon2 != -1):
        # convert to radians
        d_lat = np.radians(lat2 - lat1)
        d_lon = np.radians(lon2 - lon1)

        r_lat1 = np.radians(lat1)
        r_lat2 = np.radians(lat2)

        # argument under the root
        a = (np.sin(d_lat/2.) ** 2) + (np.cos(r_lat1) * np.cos(r_lat2) * (np.sin(d_lon/2.) ** 2))

        haversine = 2 * earth_radius * np.arcsin(np.sqrt(a))

    return haversine


def min_distance(pos, pos_list):
    # if anything lacks, it returns -1.
    distance = -1.

    for pos_ in pos_list:
        d = dist_from_coordinates(pos[0], pos[1], pos_[0], pos_[1])
        if (d < distance) or (distance == -1.):
            distance = d

    return distance


def distance_user_coupon(df, coupons_areas_dict, mean_dist=400.):
    count = 0
    todo = float(len(df) - 1)
    distance_list = []

    print('compute the (minimum) distance coupon / user...')

    if df is not None:

        # calculate the distance for each (user, coupon) couple
        for id, row in df[['USER_LATITUDE', 'USER_LONGITUDE', 'COUPON_ID_hash']].iterrows():
            # get the user/coupon properties
            user_latitude = row['USER_LATITUDE']
            user_longitude = row['USER_LONGITUDE']
            coupon_id = row['COUPON_ID_hash']

            # get the list of areas where the coupon is available
            coupon_areas = coupons_areas_dict.get(coupon_id, [])

            # compute the minimum distance user / coupon
            distance = min_distance((user_latitude, user_longitude), coupon_areas)
            if distance < 0.:
                distance = mean_dist

            # put the distance in the dataframe
            distance_list.append(distance)

            # display the progression
            count += 1
            if count % 1000 == 0:
                display_progress(100. * float(count) / todo)

        df['DISTANCE'] = pd.Series(np.array(distance_list))


def centroid(coord_list):
    mean = np.array(coord_list)
    if coord_list:
        mean = mean.mean(axis=0)
    return mean


# CATEGORIES CONVERSION ======================================================


def create_dictionary(df, column, file_path=None):
    mapping_dict = dict()
    mapping_list = []

    # get the list of values
    if df is not None:
        if type(df) == pd.core.frame.DataFrame:
            mapping_list = list(df[column].unique())
        elif type(df) == set:
            mapping_list = list(df)
        elif type(df) == list:
            mapping_list = df

    # map the list of values to an arbitrary integer
    print('[setup] \t creating a dictionary to map "{}" values to integers...'.format(column))
    for i in xrange(len(mapping_list)):
        mapping_dict[mapping_list[i]] = float(i+1)

    # define the path where the dictionary will be exported
    saving_path = file_path
    if not file_path:
        saving_path = '../input/ref/' + column + '.csv'

    # actually write the data on the HD
    print('[setup] \t writting the dictionary for "{}" in {}...'.format(column, saving_path))
    pd.DataFrame(np.array(mapping_dict.items()), columns=[column, column+'_NEW']).to_csv(saving_path, index=False)
    return mapping_dict


def load_dictionary(file_path=None, reverse=False):
    mapping_dict = dict()

    if os.path.isfile(file_path):

        # load the dictionary from HD
        mapping_df = load_dataset(file_path)
        mapping_keys = mapping_df.keys()

        # populate a dictionary with the data
        base_col = 1 if reverse else 0
        target_col = 0 if reverse else 1
        print('[setup] \t creating a dictionary from {} to {}'.format(mapping_keys[base_col], mapping_keys[target_col]))
        for row in mapping_df.iterrows():
            base_id = row[1][base_col]
            target_id = row[1][target_col]
            mapping_dict[base_id] = target_id

    return mapping_dict


def map_columns(df, columns, dictionary=None, inplace=True):
    if df is not None:
        for col in columns:

            # create the translation dictionary
            if not dictionary:
                mapping_dict = create_dictionary(df, col)
            else:
                mapping_dict = dictionary

            # either replace the column or duplicate it
            if not inplace:
                print('[mapping] \t {} => {}'.format(col, col+'_NEW'))
                df[(col+'_NEW')] = df[col].map(mapping_dict)
            else:
                print('[mapping] \t replacing {}'.format(col))
                df[col] = df[col].map(mapping_dict)


# DATAFRAME CLEANING ==========================================================


def fill_nan(df, columns=None, fill_value=-1.):
    if df is not None:
        col_list = columns
        if not col_list:
            col_list = df.keys()
        for col in col_list:
            print('[cleaning] \t replacing Nan in {} by {}...'.format(col, fill_value))
            df[col].fillna(fill_value, inplace=True)


def columns_to_float(df, columns=None):
    if df is not None:

        # if no column is specified, cast the whole dataframe
        if not columns:
            print('[cleaning] \t casting the whole dataframe to float64')
            df.astype('float64')
        # otherwise cast only the given column
        else:
            for col in columns:
                print('[cleaning] \t casting "{}" to float64...'.format(col))
                df[col] = df[col].astype('float64')


# DATAFRAME STRUCTURE =========================================================


def drop_columns(df, columns):
    if df is not None:
        for col in columns:
            print('[cleaning] \t dropping {}...'.format(col))
            df.drop(col, axis=1, inplace=True)


def keep_columns(df, columns):
    if df is not None:
        columns_to_drop = set(df.keys())
        columns_to_drop = columns_to_drop.difference(set(columns))
        columns_to_drop = list(columns_to_drop)
        drop_columns(df, columns_to_drop)


# IMPORTING / EXPORTING =======================================================


def load_dataset(file_path):
    print('[reading] \t {}'.format(file_path))
    df = pd.read_csv(file_path)

    return df


def write_dataset(df, dir_path, labels_column=None, columns_to_keep=None, columns_to_sort=None, sorting_order=None, number_split=1):
    if df is not None:

        if columns_to_sort and sorting_order:
            print('[exportation] \t sorting wrt {}'.format(columns_to_sort[0]))
            df.sort(columns=columns_to_sort, ascending=sorting_order, inplace=True, axis=0)

        # reindex the WHOLE dataframe so that x and y are still aligned
        print('[exportation] \t resetting the index')
        df.reindex(copy=False, fill_value=-1.)

        # split the dataframe in two : the data and the labels
        x_keys = columns_to_keep if columns_to_keep else df.keys()
        y_key = labels_column if labels_column else x_keys[0]
        x_df = pd.DataFrame(df[x_keys], columns=x_keys, index=df.index)
        y_df = pd.DataFrame(df[y_key], columns=[y_key], index=df.index)

        # keep only the desired properties in the exported dataframe
        keep_columns(x_df, x_keys)

        # cast everything to float
        columns_to_float(x_df)
        columns_to_float(y_df)

        # set the path to the exported files
        x_path = dir_path + 'x_{}.csv'
        y_path = dir_path + 'y_{}.csv'

        split_length = df.index.size / number_split
        for i in range(number_split):

            # list of indexes contained in split i
            i_min = i * split_length
            i_max = (i + 1) * split_length
            if i == (number_split - 1):
                i_max = df.index.size

            # write x_i
            print('[exportation]\twritting {}...'.format(x_path.format(i)))
            x_df.ix[df.index[range(i_min, i_max)]].to_csv(x_path.format(i), sep=',', index=False)

            # write y_i
            if labels_column:
                print('[exportation]\twritting {}...'.format(y_path.format(i)))
                y_df.ix[df.index[range(i_min, i_max)]].to_csv(y_path.format(i), sep=',', index=False)

    return 0
