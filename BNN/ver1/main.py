# -*- coding: utf-8 -*-

from config import Config
from data import Data
from encoder_decoder_model import EncoderDecoder

config = Config('config.json')
model_config = config.get_model_config()
data_config = config.get_data_config()

data = Data(data_config)
data.series_to_supervised('encoder', model_config['encoder_num_inputs'], model_config['encoder_num_outputs'])
data.series_to_supervised('decoder', model_config['decoder_num_inputs'], model_config['decoder_num_outputs'])


model = EncoderDecoder(model_config, data)
#model.train_model(data)

data.init_iterator('encoder', batch_size=4)
outputs = model.fit_encoder(data.next_batch('encoder', 'test')[0])