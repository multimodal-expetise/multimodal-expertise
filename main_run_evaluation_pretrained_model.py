import os

from module_test import test

if __name__ == '__main__':
    dataset_name = "Expertise"  # Dataset
    featurePath = r"data/expertise_test.pkl"  # Path to data

    model_names = ['A1_LSTM_l', 'A2_LSTM_a', 'A3_LSTM_v', 'A4_LF_LSTM_la', 'A5_LF_LSTM_lv', 'A6_LF_LSTM_av', 'A7_LF_LSTM_lav','A8_LF_MLP_lav',
                   'A9_EF_MLP_lav', 'A10_EF_LSTM_lav','A11_MFN_lav']

    for model_name in model_names:
        print("\n" + "=" * 50 , f"Testing model: {model_name}","=" * 50 + "\n")
        test(model_name, dataset_name, featurePath)
