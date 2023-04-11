import argparse

parser = argparse.ArgumentParser(description='Time-Series Data Anomaly Detection')
parser.add_argument('--dataset',
                    metavar='-d',
                    type=str,
                    required=False,
                    default='synthetic',
                    help="dataset from ['SMAP', 'MSL']")
parser.add_argument('--model',
                    metavar='-m',
                    type=str,
                    required=False,
                    default='TranAD',
                    help="model name,for now only TranAD is available")
parser.add_argument('--test',
                    action='store_true',
                    help="test the model")
parser.add_argument('--retrain',
                    action='store_true',
                    help="retrain the model")
parser.add_argument('--less',
                    action='store_true',
                    help="train using less data")
args = parser.parse_args()
