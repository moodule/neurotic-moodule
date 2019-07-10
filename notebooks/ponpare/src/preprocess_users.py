from preprocessing_lib import *


__author__ = "EelMood"


# PREPROCESSING PROCEDURE =====================================================


def preprocess_users(users_df, sex_dict, latitude_dict, longitude_dict):

    # removing Nan values
    fill_nan(users_df)

    # converting dates to timestamps...
    columns_to_timestamp(users_df, ['REG_DATE', 'WITHDRAW_DATE'], 'datetime')

    # converting the sex labels "m" and "f" to 1 and 2...
    map_columns(users_df, ['SEX_ID'], dictionary=sex_dict, inplace=True)

    # map the PREF_NAME to its gps coordinates
    print('[mapping] \t "PREF_NAME" => ("USER_LATITUDE", "USER_LONGITUDE")')
    users_df['USER_LATITUDE'] = users_df['PREF_NAME'].map(latitude_dict)
    users_df['USER_LONGITUDE'] = users_df['PREF_NAME'].map(longitude_dict)

    # removing Nan values again (created by the prefecture => (lat, long) mapping)
    fill_nan(users_df)
