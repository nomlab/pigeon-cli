import math
import glob
import json
import sys
import numpy as np
import pandas as pd
from datetime import datetime as dt

def read_csvs_in_order(filename_pattern, usecols = None, engine = 'python'):
    df = pd.DataFrame([])
    files = glob.glob(filename_pattern)
    files.sort()
    for f in files:
        df = df.append(pd.read_csv(f, usecols = usecols,
                                   engine = engine),
                       ignore_index = True)
    return df

def read_csv(f, usecols = None, engine = 'python'):
    return pd.read_csv(f, usecols = usecols, engine = engine)

def read_jsons_in_order(filename_pattern):
    data_list = []
    files = glob.glob(filename_pattern)
    files.sort()
    sys.stderr.write('Read {0} json files'.format(len(files)))
    for f in files:
        io = open(f, 'r')
        data_list.extend(json.load(io))
    return data_list

def read_json(io):
    return json.load(io)

def hash_to_df(dics, names):
    df = pd.DataFrame([])
    for name in names:
        df[name] = dics[name]
    return df


def datetime_to_json_serializer(o):
    if isinstance(o, dt):
        return o.isoformat()
    raise TypeError(repr(o) + " is not JSON serializable")

def write_json(f, dic):
    json.dump(dic, f, default = datetime_to_json_serializer)

def sort_by_time(lst, attr_name):
    parse_datetime(lst, attr_name)
    lst.sort(key = lambda dic: dic[attr_name])

def parse_datetime(lst, attr_name):
    for x in lst:
        x[attr_name] = dt.strptime(x[attr_name], '%Y-%m-%dT%H:%M:%S')

def load_data_for_train(data, data_shape = (7, 24, 3)):
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    n_prev = data_shape[0] * data_shape[1]
    for i in range(len(data) - n_prev):
        x = data[i: i + n_prev]
        x.shape = data_shape
        docX.append(x)
        docY.append(data[i + n_prev])
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def load_data_for_predict(data, data_shape = (7, 24, 3)):
    docX = []
    n_prev = data_shape[0] * data_shape[1]
    x = data[len(data) - n_prev: len(data)]
    x.shape = data_shape
    docX.append(x)
    alsX = np.array(docX)

    return alsX

def prepare_dataset(df, colnames = []):
    for colname in colnames:
        df[[colname]] -= np.min(np.abs(df[[colname]]))
        df[[colname]] /= np.max(np.abs(df[[colname]]))
    return df[colnames].values.astype('float32')

def group_by_step(df, steps):
    return [df.iloc[i:(i + steps), :] for i in range(0, len(df), steps)]

def stayprob(hems):
    power, gas, water = hems
    prob =  1 / (1 + 100 * math.exp(-3 * (power * 5.3)))
    if water > 0.3:
        prob = prob * 0.6 + 0.4
    if gas > 0.4:
        prob = prob * 0.6 + 0.4
    return prob

def calc_stayprob(lst):
    return [stayprob(x) for x in lst]
