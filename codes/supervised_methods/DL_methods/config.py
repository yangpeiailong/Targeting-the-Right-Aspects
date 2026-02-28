MODEL_CONFIGS = {
    'random_att_linear': {
            'class': 'RandomAttLinear',
            'params': {
                'embed_size': 100,
            },
            'save_path': 'model/model_random_att_linear.pkl'
    },
    'random_average_linear': {
            'class': 'RandomAverageLinear',
            'params': {
                'embed_size': 100,
            },
            'save_path': 'model/model_random_average_linear.pkl'
    },
    'random_cnn_linear': {
        'class': 'RandomCNNLinear',
        'params': {
            'embed_size': 100,
            'num_of_filters': 100,
            'window_sizes': [2, 3, 4],
        },
        'save_path': 'model/model_random_cnn_linear.pkl'
    },
    'random_lstm_att_linear': {
        'class': 'RandomLSTMAttLinear',
        'params': {
            'embed_size': 100,
            'hidden_size_lstm': 128,
        },
        'save_path': 'model/model_random_lstm_att_linear.pkl'
    },
    'random_lstm_cnn_linear': {
        'class': 'RandomLSTMCNNLinear',
        'params': {
            'embed_size': 100,
            'hidden_size_lstm': 128,
            'num_of_filters': 100,
            'window_sizes': [2, 3, 4],
        },
        'save_path': 'model/model_random_lstm_linear.pkl'
    },
    'random_lstm_linear': {
        'class': 'RandomLSTMLinear',
        'params': {
            'embed_size': 100,
            'hidden_size_lstm': 128,
        },
        'save_path': 'model/model_random_lstm_linear.pkl'
    },
    'bert_att_linear': {
        'class': 'BertAttLinear',
        'params': {
            'model': 'bert-base-chinese', #
            'embed_size': 768,  # BERT default dimension
        },
        'save_path': 'model/model_bert_att_linear.pkl'
    },
    'bert_average_linear': {
        'class': 'BertAverageLinear',
        'params': {
            'model': 'bert-base-chinese',
            'embed_size': 768,  # BERT default dimension
        },
        'save_path': 'model/model_bert_average_linear.pkl'
    },
    'bert_cls_linear': {
        'class': 'BertCLSLinear',
        'params': {
            'model': 'bert-base-chinese',
            'embed_size': 768,  # BERT default dimension
        },
        'save_path': 'model/model_bert_cls_linear.pkl'
    },
    'bert_cnn_linear': {
        'class': 'BertCNNLinear',
        'params': {
            'model': 'bert-base-chinese',
            'embed_size': 768,  # BERT default dimension
            'num_of_filters': 100,
            'window_sizes': [2, 3, 4],
        },
        'save_path': 'model/model_bert_cnn_linear.pkl'
    },
    'bert_lstm_att_linear': {
        'class': 'BertLSTMAttLinear',
        'params': {
            'model': 'bert-base-chinese',
            'embed_size': 768,  # BERT default dimension
            'hidden_size_lstm': 128,
        },
        'save_path': 'model/model_bert_lstm_att_linear.pkl'
    },
    'bert_lstm_cnn_linear': {
        'class': 'BertLSTMCNNLinear',
        'params': {
            'model': 'bert-base-chinese',
            'embed_size': 768,  # BERT default dimension
            'hidden_size_lstm': 128,
            'num_of_filters': 100,
            'window_sizes': [2, 3, 4],
        },
        'save_path': 'model/model_bert_lstm_cnn.pkl'
    },
    'bert_lstm_linear': {
        'class': 'BertLSTMLinear',
        'params': {
            'model': 'bert-base-chinese',
            'embed_size': 768,  # BERT default dimension
            'hidden_size_lstm': 128,
        },
        'save_path': 'model/model_bert_lstm_linear.pkl'
    }
}