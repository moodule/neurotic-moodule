from preprocess_areas import *
from preprocess_coupons import *
from preprocess_users import *
from preprocess_features import *

__author__ = "EelMood"


# 1) remplacer les pref_name dans le dictionnaire precedent par [ (lat,long) ] correspondant a chaque prefecture
# 2) faire la meme chose pour les utilisateurs
# 3) joindre les utilisateurs et les coupons, en calculant leur distance MINIMALE
# 4) creer le dataset Y correspondant
# 5) effectuer le meme traitement pour les donnees de test
# 6) ecrire les datasets sur le disque
# 7) creer le modele
# 8) normaliser les features ?
# 9) decouper le dataset d'entrainement en N couples aleatoires (70%, 30%)
# ameliorer la mise en forme et mettre print... done. sur une meme ligne
# regarder de plus pres pourquoi certains COUPONS_ID dans visits ne sont pas references dans coupons_train

# todo fusionner les mapping coupon_id_train et test
# todo mettre l'option de sauvegarde dans "create_dictionary"
# todo fonction qui permette de mapper c1 df1 avec c2 df2
# todo changer les iterations en iterrows, par securite
# todo inverser genre et capsulte ? (il y a plus de categories capsules)
# todo incorporer coupons_test au set d'entrainement avec les visites
# remplacer les nan par 0 (en fait une valuer quelconque, en argument optionel)

# 2) cross join users x coupons...
# TODO 3) think of something clever...
# 4) cross join users x coupons_test = test set
# TODO retirer les couples (user, coupon) pour lesquels il y a visite  ET achat : ne garder que l'achat

# test : toutes les prefectures de coupon_area_train sont bien dans prefecture_locations (y o u p i)
# users : quelle est la signification de 'withdrawn_date' ?? les utilisateurs qui ont cette valeur de renseignee
#   cintinuent a visiter le site...
# ATTENTION : pb avec les clefs de prefectures : PREF_NAME n'existe pas, utiliser pref.keys(0)
# ATTENTION : il faut convertir pref[PREF_NAME] en liste, sinon les tests ne se font pas correctement


distance_type = 2
mean_distance = 0.  # an estimation based on the first 7x10^6 coupon x users


print('=========================================================== LOADING DATA')

# input files from kaggle
users = load_dataset('../input/translated/user_list_translated.csv')
coupons_train = load_dataset('../input/translated/coupon_list_train_translated.csv')
coupons_test = load_dataset('../input/translated/coupon_list_test_translated.csv')
visits = load_dataset('../input/coupon_visit_train.csv')
transactions = load_dataset('../input/translated/coupon_detail_train_translated.csv')
prefectures = load_dataset('../input/translated/prefecture_locations_translated.csv')
coupons_area_train = load_dataset('../input/translated/coupon_area_train_translated.csv')
coupons_area_test = load_dataset('../input/translated/coupon_area_test_translated.csv')

# conversion dictionaries
user_id_mapping = load_dictionary('../input/ref/USER_ID_hash.csv')
coupon_id_mapping = load_dictionary('../input/ref/COUPON_ID_hash.csv')
capsule_mapping = load_dictionary('../input/ref/CAPSULE_TEXT.csv')
genre_mapping = load_dictionary('../input/ref/GENRE_NAME.csv')
sex_mapping = load_dictionary('../input/ref/SEX_ID.csv')

print('done.')


print('===================================================== SETUP REFERENTIALS')

# extract the unique ids of the data objects
print('[setup] \t extracting the lists of unique ids')
users_id_set = set(users['USER_ID_hash'].unique())
coupons_id_set = set(coupons_train['COUPON_ID_hash'].unique())          # init with the ids from the training set
coupons_id_set.update(set(coupons_test['COUPON_ID_hash'].unique()))     # add the ids of the testing coupons

# check whether the dictionaries where loaded ; if not create them
if not user_id_mapping:
    user_id_mapping = create_dictionary(users, column='USER_ID_hash')
if not coupon_id_mapping:
    coupon_id_mapping = create_dictionary(coupons_id_set, column='COUPON_ID_hash')
if not capsule_mapping:
    capsule_mapping = create_dictionary(coupons_train, column='CAPSULE_TEXT')
if not genre_mapping:
    genre_mapping = create_dictionary(coupons_train, column='GENRE_NAME')
if not sex_mapping:
    sex_mapping = create_dictionary(users, column='SEX_ID')

print('done.')


print('==================================================== PREPROCESSING AREAS')

# create the mapping from prefectures to gps coordinates
latitude_mapping, longitude_mapping = prefectures_to_gps(prefectures)

# create the mapping coupon => available areas (in gps)
if distance_type == 1:
    coupons_area_dict = areas_to_gps(coupons_area_train,
                                     latitude_dict=latitude_mapping,
                                     longitude_dict=longitude_mapping)
    coupons_area_dict.update(areas_to_gps(coupons_area_test,
                                          latitude_dict=latitude_mapping,
                                          longitude_dict=longitude_mapping))
else:
    # create the mapping coupon => ken_name gps
    coupons_area_dict = coupons_to_gps(coupons_train,
                                       latitude_dict=latitude_mapping,
                                       longitude_dict=longitude_mapping)
    coupons_area_dict.update(coupons_to_gps(coupons_test,
                                            latitude_dict=latitude_mapping,
                                            longitude_dict=longitude_mapping))

print('done.')


print('================================================== PREPROCESSING COUPONS')

# preprocess the coupon dataframes
preprocess_coupons(coupons_train, capsule_dict=capsule_mapping, genre_dict=genre_mapping)
preprocess_coupons(coupons_test, capsule_dict=capsule_mapping, genre_dict=genre_mapping)

print('done.')


print('==================================================== PREPROCESSING USERS')

# preprocess the users dataframe
preprocess_users(users, sex_dict=sex_mapping, latitude_dict=latitude_mapping, longitude_dict=longitude_mapping)

print('done.')


print('=================================================== PREPROCESSING VISITS')

# define a procedure to check if a coupon is in the datasets


def coupon_is_known(coupon_hash):
    res = -1.
    if coupon_hash in coupons_id_set:
        res = 1.
    return res

# remove the records whose coupon_id is not know
print('[cleaning ]remove the visit\'s records whose "COUPON_ID_hash" is unknown')
visits['TEST'] = visits['VIEW_COUPON_ID_hash'].map(coupon_is_known)
visits_clean = visits[visits['TEST'] > 0]

print('done.')


print('============================================= PREPROCESSING TRANSACTIONS')

# convert dates to timestamps
columns_to_timestamp(df=transactions, columns=['I_DATE'], origin='datetime')

# convert int64 to float64
columns_to_float(transactions, ['ITEM_COUNT'])

print('done.')


print('=========================================== JOINING THE TESTING DATASETS')
# it is the cross join of users and coupons_test


# adding a common column to users and coupons to do a cross join
print('add a "COMMON_ID" column both to users and coupons_test to perform a cross join...')
users['COMMON_ID'] = pd.DataFrame(np.array(len(users)*[1]))
coupons_test['COMMON_ID'] = pd.DataFrame(np.array(len(coupons_test)*[1]))


# cross join users and coupons test
print('cross join users and coupons_test...')
x_post = pd.merge(users, coupons_test, how='outer', on=['COMMON_ID'])


print('done.')


print('============================== JOINING THE TRAINING DATASETS WITH VISITS')
# done according to the visits table


# merge with users
print('merging users and visits...')
x_visits = pd.merge(visits_clean, users, how='inner', on=['USER_ID_hash'])


# merge with coupons
print('merging coupons and visits...')
x_visits.rename(columns={'VIEW_COUPON_ID_hash': 'COUPON_ID_hash'}, inplace=True)
x_visits = pd.merge(x_visits, coupons_train, how='inner', on=['COUPON_ID_hash'])


print('done.')


print('======================== JOINING THE TRAINING DATASETS WITH TRANSACTIONS')


print('merging users and transactions...')
x_transactions = pd.merge(transactions, users, how='inner', on=['USER_ID_hash'])


print('merging users and transactions...')
x_transactions = pd.merge(x_transactions, coupons_train, how='inner', on=['COUPON_ID_hash'])


print('adding a "PURCHASE_FLG" column to x_transactions...')
x_transactions['PURCHASE_FLG'] = pd.Series(np.array(len(x_transactions)*[1]))


print('append (-twice-) as much samples with y=0...')
rows = np.random.choice(x_visits.index, 1 * len(x_transactions))
x_ = x_visits.ix[rows]
x_ = x_[x_['PURCHASE_FLG'] == 0]
x_transactions = x_transactions.append(x_, ignore_index=True)


print('done.')


print('==================================================== FEATURE ENGINEERING')


# dictionary to hold the bought coupons by each user
print('creating the dictionary of the coupons\' centroids')
coupons_centroids = dict()
for coupon_id in list(coupons_id_set):
    coupons_centroids[coupon_id] = []
print('creating the dictionary of the users\' centroids')
users_centroids_latitude = dict()
users_centroids_longitude = dict()
users_purchases = dict()
for user_id in list(users_id_set):
    users_centroids_latitude[user_id] = -1.
    users_centroids_longitude[user_id] = -1.
    users_purchases[user_id] = []


# centroids of the areas of availability for each coupon
print('computing the validity centroid of each coupon...')
for coupon_id in list(coupons_id_set):
    coord_list = coupons_area_dict.get(coupon_id, [])
    coupons_centroids[coupon_id] = list(centroid(coord_list))


# list of the centroids of the coupons bought by each user
print('deducing the centroid of the purchases made by each user...')
for index, row in x_transactions[['USER_ID_hash', 'COUPON_ID_hash']].iterrows():
    user_id = row['USER_ID_hash']
    coupon_id = row['COUPON_ID_hash']
    temp = users_purchases[user_id]
    coupon_centroid = coupons_centroids.get(coupon_id, [])
    if coupon_centroid:
        temp.append(coupon_centroid)
    users_purchases[user_id] = temp

# making a mpping dictionary out of it
print('creating a mapping dictionary out of it...')
for user_id, coord_list in users_purchases.iteritems():
    user_location = list(centroid(coord_list))
    if user_location:
        users_centroids_latitude[user_id] = user_location[0]
        users_centroids_longitude[user_id] = user_location[1]

print('done.')


print('================================================== CLEANING THE DATASETS')


print('mapping purchase centroids to users...')
x_post['USER_LATITUDE'] = x_post['USER_ID_hash'].map(users_centroids_latitude)
x_post['USER_LONGITUDE'] = x_post['USER_ID_hash'].map(users_centroids_longitude)


# compute the minimum distances for all (user, coupon) in the datasets
distance_user_coupon(x_visits, coupons_area_dict, mean_distance)
distance_user_coupon(x_post, coupons_area_dict, mean_distance)
distance_user_coupon(x_transactions, coupons_area_dict, mean_distance)


# convert the "VISIT_DATE" to timestamp
# columns_to_timestamp(x_visits, ['I_DATE'], 'datetime')


# convert hashed ids to integer ids
map_columns(x_post, ['USER_ID_hash'], user_id_mapping, inplace=True)
map_columns(x_visits, ['USER_ID_hash'], user_id_mapping, inplace=True)
map_columns(x_transactions, ['USER_ID_hash'], user_id_mapping, inplace=True)
map_columns(x_post, ['COUPON_ID_hash'], coupon_id_mapping, inplace=True)
map_columns(x_visits, ['COUPON_ID_hash'], coupon_id_mapping, inplace=True)
map_columns(x_transactions, ['COUPON_ID_hash'], coupon_id_mapping, inplace=True)


# inplace the user_id column to use it as a training feature
x_post['USER_ID_hash_NEW'] = x_post['USER_ID_hash']
x_visits['USER_ID_hash_NEW'] = x_visits['USER_ID_hash']
x_transactions['USER_ID_hash_NEW'] = x_transactions['USER_ID_hash']


print('done.')


print('=================================================== WRITING THE DATASETS')


# split the features list to try and select those used to train the models
id_columns = ['USER_ID_hash', 'COUPON_ID_hash']
engineered_columns = ['DISTANCE', 'USABILITY_SCORE']
user_columns = ['USER_ID_hash_NEW', 'AGE', 'SEX_ID', 'USABLE_DATE_HOLIDAY', 'USABLE_DATE_BEFORE_HOLIDAY']
coupon_columns = ['GENRE_NAME', 'DISCOUNT_PRICE', 'PRICE_RATE', 'DISPPERIOD', 'VALIDPERIOD']


# redundant or less informative features
user_secondary_columns = ['REG_DATE', 'WITHDRAW_DATE', 'USER_LATITUDE', 'USER_LONGITUDE']
coupon_secondary_columns = ['CAPSULE_TEXT', 'CATALOG_PRICE', 'DISPFROM', 'DISPEND', 'VALIDFROM', 'VALIDEND']
usable_columns = ['USABLE_DATE_MON', 'USABLE_DATE_TUE', 'USABLE_DATE_WED', 'USABLE_DATE_THU',
                  'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN']


# write visits dataset on the HD
write_dataset(df=x_visits,
              dir_path='../data/visits/',
              labels_column='PURCHASE_FLG',
              columns_to_keep=engineered_columns+user_columns+coupon_columns,
              columns_to_sort=['COUPON_ID_hash', 'DISTANCE'],
              sorting_order=[1, 1])

# write transaction dataset on HD
write_dataset(df=x_transactions,
              dir_path='../data/detail_50/',
              labels_column='PURCHASE_FLG',
              columns_to_keep=engineered_columns+user_columns+coupon_columns,
              columns_to_sort=['COUPON_ID_hash', 'DISTANCE'],
              sorting_order=[1, 1])

# write post dataset on HD
write_dataset(df=x_post,
              dir_path='../data/post/',
              columns_to_keep=id_columns+engineered_columns+user_columns+coupon_columns,
              columns_to_sort=['COUPON_ID_hash', 'DISTANCE'],
              sorting_order=[1, 1],
              number_split=16)

# write test dataset
write_dataset(df=x_post,
              dir_path='../data/',
              columns_to_keep=['COUPON_ID_hash', 'USER_ID_hash', 'DISTANCE', 'PRICE_RATE', 'USABILITY_SCORE'],
              columns_to_sort=['DISTANCE', 'PRICE_RATE', 'USABILITY_SCORE'],
              sorting_order=[1, 0, 0])

print('done.')
