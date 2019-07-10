from preprocessing_lib import *


__author__ = 'EelMood'


# TODO faire un essai en ne prenant que les coupons pour lesquels p>0.5 et un autre avec systematiquement 10 coupons


# define the script arguments
training_set = 'detail'
model = 'mlp_28x28x21x14x2'


print('============================================== CREATING SUBMISSION TABLE')


print('[{}] \t reading users csv file'.format(1))
users = pd.read_csv('../input/translated/user_list_translated.csv')


print('[{}] \t initializing table'.format(2))
user_predictions = dict()
for user_hash in list(users['USER_ID_hash'].unique()):
    user_predictions[user_hash] = []


print('done.')


print('=================================================== LOADING DICTIONARIES')

# read dictionaries to revert the integer ids back to hashed strings
user_id_mapping = load_dictionary('../input/ref/USER_ID_hash.csv', reverse=True)
coupon_id_mapping = load_dictionary('../input/ref/COUPON_ID_hash.csv', reverse=True)

print('done.')


print('==================================================== LOADING PREDICTIONS')


# predictions_list = []
#
# for i in range(16):
#
#     print('[{}] \t reading csv file'.format(i))
#     predictions = pd.read_csv('./results/' + training_set + '/' + model + '/predictions_' + str(i) + '.csv')
#
#     print('[{}] \t mapping integer ids to hash ids'.format(i))
#     predictions['USER_ID'] = predictions['USER_ID'].map(user_id_mapping)
#     predictions['COUPON_ID'] = predictions['COUPON_ID'].map(coupon_id_mapping)
#
#     print('[{}] \t concatenating dataframes'.format(i))
#     predictions_list.append(predictions)
#
#
# print('[{}] \t sorting predictions by likelihood'.format(16))
# all_predictions = pd.concat(predictions_list, axis=0, ignore_index=True)
# all_predictions.sort(columns=['LIKELIHOOD'], inplace=True, axis=0, ascending=False) # test : it should be False !

all_predictions = pd.read_csv('../data/distance_1.csv')
map_columns(all_predictions, columns=['USER_ID_hash'], dictionary=user_id_mapping, inplace=True)
map_columns(all_predictions, columns=['COUPON_ID_hash'], dictionary=coupon_id_mapping, inplace=True)


print('done.')


print('============================================== FORMATTING FOR SUBMISSION')

print('[{}] \t filling the submission table'.format(1))
count = 1
for id, row in all_predictions.iterrows():
    user_hash = row['USER_ID_hash']
    coupon_hash = row['COUPON_ID_hash']
    likelihood = row['DISTANCE']
    temp = user_predictions[user_hash]
    if len(temp) < 100:
        temp.append(coupon_hash)
        user_predictions[user_hash] = temp
    count = count + 1
    if count % 500000 == 0:
        print('[{}] \t {}...'.format(1, count))


print('[{}] \t concatenating the coupon hash'.format(2))
submission = dict()
for user_hash, coupon_list in user_predictions.iteritems():
    temp = " ".join(coupon_list)
    submission[user_hash] = temp


print('[{}] \t writting the submission file'.format(3))
submission_df = pd.DataFrame(np.array(submission.items()), columns=['USER_ID_hash', 'PURCHASED_COUPONS'])
submission_df.to_csv('./results/' + training_set + '/' + model + '/submission.csv', index=False)


print('done.')
