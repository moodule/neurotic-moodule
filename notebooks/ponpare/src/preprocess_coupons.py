from preprocessing_lib import *


__author__ = "EelMood"


# PREPROCESSING PROCEDURE  ====================================================

def preprocess_coupons(coupons_df, capsule_dict, genre_dict):

    # map categories to intger ids
    map_columns(coupons_df, ['CAPSULE_TEXT'], dictionary=capsule_dict, inplace=True)
    map_columns(coupons_df, ['GENRE_NAME'], dictionary=genre_dict, inplace=True)

    # convert dates to timestamps
    columns_to_timestamp(coupons_df, ['VALIDFROM', 'VALIDEND'], 'date')
    columns_to_timestamp(coupons_df, ['DISPFROM', 'DISPEND'], 'datetime')

    # sum the "USABLE_XXX" columns to get a new feature = "USABILITY_SCORE"
    keys = ['USABLE_DATE_MON', 'USABLE_DATE_TUE', 'USABLE_DATE_WED', 'USABLE_DATE_THU']
    keys = keys + ['USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN']
    print('[feature] \t calculating a global usability score')
    coupons_df['USABILITY_SCORE'] = coupons_df[keys].sum(axis=1)

    # replace the Nan in case the previous processings have generate some new ones
    fill_nan(coupons_df)
