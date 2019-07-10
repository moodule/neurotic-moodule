__author__ = 'EelMood'

import os, sys
import numpy as np
import pandas as pd
import goslate as gt

capsule = pd.read_csv('../input/ref/capsule.csv')
translation_map = {k:v for (k,v) in zip(capsule['CAPSULE_TEXT'], capsule['English Translation'])}

all_keys = np.array(list(translation_map.keys()))

gs = gt.Goslate()

for k,v in translation_map.items():
     gv =gs.translate(k,'en',source_language='ja')
     if not (v == gv):
         print ('Test translate: {} should be [{}] but auto-translated to [{}]'.format(k,v,gv))

todo = {
     '../input/coupon_area_test.csv'         :['SMALL_AREA_NAME','PREF_NAME'],
     '../input/coupon_area_train.csv'        :['SMALL_AREA_NAME','PREF_NAME'],
     '../input/coupon_detail_train.csv'      :['SMALL_AREA_NAME'],
     '../input/coupon_list_test.csv'         :['CAPSULE_TEXT','GENRE_NAME','ken_name','large_area_name','small_area_name'],
     '../input/coupon_list_train.csv'        :['CAPSULE_TEXT','GENRE_NAME','ken_name','large_area_name','small_area_name'],
     '../input/prefecture_locations.csv'     :['PREF_NAME','PREFECTUAL_OFFICE'],
     # '../input/coupon_visit_train.csv'       :[],
     '../input/user_list.csv'                :['PREF_NAME'],
}

for f,cols in todo.items():
    print ('==============================================')
    print('Reading ', f)
    infile = pd.read_csv(f)
    print ('Enriching dictionary')
    for c in cols:
        if f=='../input/prefecture_locations.csv' and c=='PREF_NAME':       #some weird ass bug : PREF_NAME column is somehow unknown...
            to_add = infile.ix[:,0].unique()
        else:
            to_add = infile[c].unique()
        if pd.isnull(to_add[0]) : to_add = to_add[1:]
        all_keys = np.union1d(all_keys, to_add)

auto_translation_map = {k:translation_map.get(k,gs.translate(k,'en','ja')) for k in all_keys}

for f,cols in todo.items():
    print ('==============================================')
    print('Mapping translation for: ', f)
    infile = pd.read_csv(f)
    for c in cols:
        if f=='../input/prefecture_locations.csv' and c=='PREF_NAME':
            infile.ix[:,0] = infile.ix[:,0].map(auto_translation_map)
        else:
            infile[c] = infile[c].map(auto_translation_map)
        infile.to_csv(f.replace(".csv", "_translated.csv").replace('../input/', '../input/translated/'), index=False)

print ('Done.')
