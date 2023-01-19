from torch import nn


def get_config(language_model):
    if language_model == 'bert':
        model_name = "bert-base-uncased"
        bert_features = 768,
        activation_func = nn.Tanh()
    elif language_model == "clinical_longformer":
        model_name = "yikuan8/Clinical-Longformer"
        bert_features = 768,
        activation_func = nn.Tanh()
    elif language_model == "roberta":
        model_name = "roberta-base"
        bert_features = 768,
        activation_func = nn.Tanh()
    elif language_model == "clinical_bert":
        model_name = "emilyalsentzer/Bio_ClinicalBERT"
        bert_features = 768,
        activation_func = nn.Tanh()
    else:
        print('supported models: bert, clinical_longformer, roberta')
        return 'error'

    return model_name, bert_features, activation_func
