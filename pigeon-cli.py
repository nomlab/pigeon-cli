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

def json2csv(args):
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
    data = {'hems_id': dics['hems_id'],
            'lat': dics['lat'],
            'lng': dics['lng'],
            'probability': probs}
    utils.write_json(args.output, data)

parser = argparse.ArgumentParser(
    prog = 'pred-hems.py',
    usage = 'pigeon-cli <command> --help',
    description = 'pigeon-cli is command line tool for predicting hems using Keras.',
    add_help = True)

subparser = parser.add_subparsers(title = 'SubCommands',
                                  description = '...',
                                  help = '...')

parser_train = subparser.add_parser('train',
                                    usage = 'pigeon-cli train []',
                                    help = 'train data and save model and weights')
parser_train.add_argument('file', nargs = '?',
                          help = 'Specify input dir for train',
                          type = argparse.FileType('r'),
                          default = sys.stdin)
parser_train.add_argument('-o', '--output',
                          help = 'Specify h5 file to output model',
                          action = 'store', default = 'model.h5')
parser_train.add_argument('-e', '--epoch',
                          help = 'Specify the number of epochs when train',
                          action = 'store', type = int, default = 100)
parser_train.add_argument('-b', '--batch',
                          help = 'Specify the number of batches',
                          action = 'store', type = int, default = 500)
parser_train.set_defaults(func = train)

parser_predict = subparser.add_parser('predict', help = 'predict hems data from trained model')
parser_predict.add_argument('file', nargs = '?',
                            help = 'Specify input file for predict',
                            type = argparse.FileType('r'),
                            default = sys.stdin)
parser_predict.add_argument('model',
                            help = 'Specify h5 file to load trained model',
                            action = 'store')
parser_predict.add_argument('n_pred',
                            help = 'Input number how long to predict in sequential',
                            type = int)
parser_predict.add_argument('-o', '--output',
                            help = 'Specify json file to output predicted data',
                            type = argparse.FileType('w'),
                            default = sys.stdout)
parser_predict.set_defaults(func = predict)

parser_json2csv = subparser.add_parser('json2csv', help = 'Convert hems data to csv from json')
parser_json2csv.add_argument('hems_id',
                             help = 'Input hems id you want to covert')
parser_json2csv.add_argument('-d', '--dir',
                             help = 'Specify input dir for convert',
                             action = 'store', required = True)
parser_json2csv.add_argument('-o', '--output',
                             help = 'Specify json file to output predicted data',
                             type = argparse.FileType('w'),
                             default = sys.stdout)
parser_json2csv.add_argument('--lat',
                             default = 10.0, type = float)
parser_json2csv.add_argument('--lng',
                             default = 10.0, type = float)
parser_json2csv.set_defaults(func = json2csv)

parser_stayprob = subparser.add_parser('stayprob', help = 'Calclate Stay Probability from energy usages')
parser_stayprob.add_argument('file', nargs = '?',
                             help = 'The csv file has power_usage, gas_usage and water_usage per hour',
                             type = argparse.FileType('r'),
                             default = sys.stdin)
parser_stayprob.add_argument('-o', '--output',
                             help = 'File to output Stay Probability',
                             type = argparse.FileType('w'),
                             default = sys.stdout)
parser_stayprob.set_defaults(func = stayprob)

args = parser.parse_args()

args.func(args)
