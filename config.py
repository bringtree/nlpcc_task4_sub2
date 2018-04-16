import configparser

config = configparser.ConfigParser()
config.read('config.ini')
train_args = {}
for key,value in config.items('train_args'):
    train_args[key] = value

for key,value in config.items('save_model'):
    model_src = value

