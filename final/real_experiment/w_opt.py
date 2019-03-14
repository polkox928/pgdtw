import libdtw as lib
from datetime import datetime
import pickle

raw_data = lib.load_data(n_to_keep=1000)
_ = raw_data.pop('reference')

data_2016 = dict((k, v) for k, v in raw_data.items() if datetime.strptime(v[0]['start'][:4], '%Y') == datetime.strptime('2016', '%Y'))

data = dict((k,v) for k,v in data_2016.items() if len(v[0]['values']) <= 708)

train_size = 101
train_data = dict(sorted(list(data.items()))[:train_size])
raw_test_data = dict(sorted(list(data.items()))[train_size:])
print("Train data: %d batches\nTest data: %d batches"%(len(train_data), len(raw_test_data)))

if 'reference' in train_data: _=train_data.pop('reference')
train_data = lib.assign_ref(train_data)
D = lib.Dtw(json_obj=train_data)
print('Number of PVs used : %d'%len(D.pv_names))

pv_ref = D.pv_names
test_data = dict()
for k, v in raw_test_data.items():
    batch = list(filter(lambda x: x['name'] in pv_ref, v))
    if len(batch) == len(pv_ref):
        test_data[k]=batch
print('Number of test batches after filtering for PVs: %d'%len(test_data))

print('Global P max: %d'%D.get_global_p_max())

if 'reference' in train_data: ref = train_data.pop('reference')
np.random.seed(42)
first_sample = np.random.choice(list(D.data['queriesID']), size=10, replace=False)
second_sample = np.random.choice(list(D.data['queriesID']), size=10, replace=False)

train_weight1 = dict((k,train_data[k]) for k in first_sample)
train_weight2 = dict((k,train_data[k]) for k in second_sample)

train_weight1['reference'] = ref
train_weight1[ref] = train_data[ref]

train_weight2['reference'] = ref
train_weight2[ref] = train_data[ref]

print('First sample')
D1 = lib.Dtw(train_weight1)
print('\nSecond sample')
D2 = lib.Dtw(train_weight2)

num_queries = D1.data['num_queries']
step_pattern = 'symmetric2'
file_path = 'data\\optWeights1.pickle'
n_jobs=-2
try:
    with open(file_path, 'rb') as f:
        D1.data['feat_weights'] = pickle.load(f)
    #print('Initial weights:\n', D.data['feat_weights'])
except OSError as ex:
#    pass
#finally:
    D1.optimize_weights(step_pattern, n_steps = 20, file_path = file_path, n_jobs=n_jobs)
    with open(file_path, 'wb') as f:
        pickle.dump(D1.data['feat_weights'], f, protocol=pickle.HIGHEST_PROTOCOL)
