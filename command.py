import os
import sys
import json
import argparse
from classifier import Classifier


def get_network_config(config_path):
    with open(config_path, 'r') as f:
        classifier_config = json.load(f)

    return classifier_config

def train(args):
  kwargs = {}
  classifier_config = get_network_config(args.config)
  classifier = Classifier(classifier_config)
  kwargs["development"] = args.development
  classifier.train(**kwargs)

def evaluate(args):
  kwargs = {}
  classifier_config = get_network_config(args.config)
  classifier = Classifier(classifier_config)
  kwargs["model_path"] = args.model_path
  kwargs["model_weights_path"] = os.path.join(os.path.dirname(args.model_path),'weights_complete.h5')
  kwargs["benchmark"] = args.benchmark
  classifier.evaluate(**kwargs)

def predict_class(args):
  kwargs = {}
  classifier_config = get_network_config(args.config)
  classifier = Classifier(classifier_config)
  kwargs["model_path"] = args.model_path
  kwargs["model_weights_path"] = os.path.join(os.path.dirname(args.model_path),'weights_complete.h5')
  kwargs["test_dir"] = args.test_dir
  classifier.predict_class(**kwargs)

def predict_pipeline(args):
  kwargs = {}
  classifier_config = get_network_config(args.config)
  classifier = Classifier(classifier_config)
  kwargs["model_path"] = args.model_path
  kwargs["model_weights_path"] = os.path.join(os.path.dirname(args.model_path),'weights_complete.h5')
  kwargs["test_dir"] = args.test_dir
  kwargs["acc"] = args.acc
  classifier.predict_pipeline(**kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train classifier')
    subparsers = parser.add_subparsers(help='sub-command help')
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('-c', '--config', required=True, help='path for the config json')

    parser_train = subparsers.add_parser('train', description='Train the model.', help='train help', parents=[base_parser])
    parser_train.add_argument('-d', '--development', help='switch to development mode', type=bool, default=False)
    parser_train.set_defaults(func=train)

    parser_evaluate = subparsers.add_parser('evaluate', description='Evaluate the model.', help='evaluate help', parents=[base_parser])
    parser_evaluate.add_argument('model_path', help='trained model')
    parser_evaluate.add_argument('-b', '--benchmark', help='evaluate on benchmark annotations', default=0)
    parser_evaluate.set_defaults(func=evaluate)

    parser_predict_class = subparsers.add_parser('predict_class', description='Predict building type and saveresults', parents=[base_parser])
    parser_predict_class.add_argument('model_path', help='trained model')
    parser_predict_class.add_argument('test_dir', help='dir with images to be predicted')
    parser_predict_class.set_defaults(func=predict_class)

    args = parser.parse_args(sys.argv[1:])
    args.func(args)
