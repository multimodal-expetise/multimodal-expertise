import os

from module_test import test

if __name__ == '__main__':
    dataset_name = "Expertise"
    featurePath = os.path.join(os.path.dirname(__file__), "data", "expertise_test.pkl")

    model_names = [
        # Single modality
        'A1_LSTM_l', 'A2_LSTM_a', 'A3_LSTM_v',
        # Dual modality
        'A4_LF_LSTM_la', 'A5_LF_LSTM_lv', 'A6_LF_LSTM_av',
        # Tri-modal fusion
        'A7_LF_LSTM_lav', 'A8_LF_MLP_lav',
        'A9_EF_MLP_lav', 'A10_EF_LSTM_lav',
        # MFN
        'A11_MFN_without_av',   # without A<->V
        'A12_MFN_without_ta',   # without T<->A
        'A13_MFN_without_tv',   # without T<->V
        'A14_MFN_lav',
    ]

    for model_name in model_names:
        print("\n" + "=" * 50, f"Testing model: {model_name}", "=" * 50 + "\n")
        test(model_name, dataset_name, featurePath)
