import sys
import argparse
from datetime import datetime as dt
from datetime import timedelta
import utils
from model import Model
from hems  import Hems, HemsCollection

def train(args):
    dics = utils.read_json(args.file)
    dataframe = utils.hash_to_df(dics['values'], names = ['timestamp', 'power_usage', 'gas_usage', 'water_usage'])
    dataframe.sort_values(by = 'timestamp')
    dataset = utils.prepare_dataset(dataframe, colnames = ["power_usage",
                                                           "gas_usage",
                                                           "water_usage"])
    data_shape = (7, 24, 3)
    (X_train, Y_train) = utils.load_data_for_train(dataset, data_shape)

    model = Model.create_from_desc(data_shape)
    model.train(X_train, Y_train, batch_size = args.batch,
                epochs = args.epoch, validation_split = 0.05)
    model.save(args.output)

def predict(args):
    dics = utils.read_json(args.file)
    dataframe = utils.hash_to_df(dics['values'], names = ['timestamp', 'power_usage', 'gas_usage', 'water_usage'])
    dataframe.sort_values(by = 'timestamp')
    dataset = utils.prepare_dataset(dataframe, colnames = ["power_usage",
                                                           "gas_usage",
                                                           "water_usage"])
    data_shape = (7, 24, 3)
    x = utils.load_data_for_predict(dataset, data_shape)
    model = Model.load(args.model)
    first_dt = dt.strptime(dataframe["timestamp"].values[-1],
                           '%Y-%m-%dT%H:%M:%S') + timedelta(hours = 1)
    predicted = model.predict_sequences(x[0], first_dt, args.n_pred, data_shape)
    predicted_hems = {'hems_id': dics['hems_id'],
                      'lat': dics['lat'],
                      'lng': dics['lng'],
                      'values': predicted}
    utils.write_json(args.output, predicted_hems)

def usages(args):
    dics = utils.read_jsons_in_order(args.dir + "/*.json")
    utils.sort_by_time(dics, 'frameworx:date')
    hems_lst = HemsCollection(dics, args.hems_id, args.lat, args.lng)
    json = hems_lst.to_hash(['timestamp', 'power_usage', 'gas_usage', 'water_usage'])
    utils.write_json(args.output, json)

def stayprob(args):
    dics = utils.read_json(args.file)
    dataframe = utils.hash_to_df(dics['values'], names = ['timestamp', 'power_usage', 'gas_usage', 'water_usage'])
    stayprobs = []
    steps = 7 * 24 # a week
    hems_per_week = utils.group_by_step(dataframe, steps)
    for hems_df in hems_per_week:
        dataset = utils.prepare_dataset(hems_df, colnames = ['power_usage', 'gas_usage', 'water_usage'])
        stayprobs.extend(utils.calc_stayprob(dataset))
    probs = [{'timestamp': dt, 'val': prob} for [dt, prob] in zip(dataframe["timestamp"].values, stayprobs)]
    data = [{'hems_id': dics['hems_id'],
            'lat': dics['lat'],
            'lng': dics['lng'],
            'probability': probs}]
    utils.write_json(args.output, data)

parser = argparse.ArgumentParser(
    prog = 'pred-hems.py',
    usage = 'python pigeon-cli.py',
    description = 'pigeon-cli is command line tool for predicting hems using Keras.',
    add_help = True)

subparser = parser.add_subparsers(title = 'SubCommands',
                                  description = '...',
                                  help = '...')

parser_train = subparser.add_parser('train',
                                    help = 'Train data and save trained model')
parser_train.add_argument('file', nargs = '?',
                          help = 'Input file (default is stdin)',
                          type = argparse.FileType('r'),
                          default = sys.stdin)
parser_train.add_argument('-o', '--output',
                          help = 'Output h5 file (default is \'model.h5\')',
                          action = 'store', default = 'model.h5')
parser_train.add_argument('-e', '--epoch',
                          help = 'The number of training epochs (default is 100)',
                          action = 'store', type = int, default = 100)
parser_train.add_argument('-b', '--batch',
                          help = 'The number of batches (default is 500)',
                          action = 'store', type = int, default = 500)
parser_train.set_defaults(func = train)

parser_predict = subparser.add_parser('predict',
                                      help = 'Predict data from trained model')
parser_predict.add_argument('file', nargs = '?',
                            help = 'Input file (default is stdin)',
                            type = argparse.FileType('r'),
                            default = sys.stdin)
parser_predict.add_argument('model',
                            help = 'h5 file to load trained model (default is \'model.h5\')',
                            action = 'store', default = 'model.h5')
parser_predict.add_argument('n_pred',
                            help = 'The number of predict in sequence',
                            type = int)
parser_predict.add_argument('-o', '--output',
                            help = 'Output file (default is stdout)',
                            type = argparse.FileType('w'),
                            default = sys.stdout)
parser_predict.set_defaults(func = predict)

parser_usages = subparser.add_parser('usages',
                                     help = 'Transform to energy usages from hems')
parser_usages.add_argument('hems_id',
                           help = 'Hems id you want to transform')
parser_usages.add_argument('-d', '--dir',
                           help = 'Directory has target hems data to transform (default is \'json\')',
                           action = 'store', default = 'json')
parser_usages.add_argument('-o', '--output',
                           help = 'Output file (default is stdout)',
                           type = argparse.FileType('w'),
                           default = sys.stdout)
parser_usages.add_argument('--lat',
                           help = 'Latitude corresponding to hems id',
                           default = 10.0, type = float)
parser_usages.add_argument('--lng',
                           help = 'Longitude corresponding to hems id',
                           default = 10.0, type = float)
parser_usages.set_defaults(func = usages)

parser_stayprob = subparser.add_parser('stayprob',
                                       help = 'Calclate Stay Probability from energy usages')
parser_stayprob.add_argument('file', nargs = '?',
                             help = 'Input file (default is stdin)',
                             type = argparse.FileType('r'),
                             default = sys.stdin)
parser_stayprob.add_argument('-o', '--output',
                             help = 'Output file (default is stdout)',
                             type = argparse.FileType('w'),
                             default = sys.stdout)
parser_stayprob.set_defaults(func = stayprob)

args = parser.parse_args()

args.func(args)
