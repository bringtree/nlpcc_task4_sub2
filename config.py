import configparser

config = configparser.ConfigParser()
config.read('config.ini')
train_args = {}
train_args["embedding_words_num"] = 11863
train_args["vec_size"] = 300
train_args["batch_size"] = 20
train_args["time_step"] = 30
train_args["sentences_num"] = 30
train_args["intents_type_num"] = 12
train_args["hidden_num"] = 200
train_args["enable_embedding"] = False
train_args["iterations"] = 100
train_args["train_output_keep_prob"] = 0.5
train_args["test_output_keep_prob"] = 1
train_args["learning_rate"] = 0.001
train_args["decay_rate"] = 0.795
train_args["decay_steps"] = 180

model_file = "./model_fasttext_200_ltp"