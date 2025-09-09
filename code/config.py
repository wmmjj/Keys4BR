import torch

K = 2
GET_CONTEXT = True
context_n = 2
get_before = True
get_after = True
remove_stack_trace = False

# task = 'severity'
# input_data_filepath = './input_data/' + task
# train_data_file = input_data_filepath + '_train.csv'
# valid_data_file = input_data_filepath + '_valid.csv'
# test_data_file = input_data_filepath + '_test.csv'
# num_labels = 2
# label_map = {'non-severity': 0, 'severity': 1} if task == 'severity' else {'low': 0, 'medium': 1, 'high': 2}
epochs = 10
lr = 1e-3
batch_size = 8
# saved_model_path = './fine-tuned-models/' + task + '_model.pt'
# predict_result_path = './predict_res/' + task + '_result.csv'

hidden_size = 768
CNN_OUTPUT_CHANNELS = 768

init_seed = 113
reproducibility = True
key_sentence_max_num = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

task1_type = "sentence_classification"
task2_type = "token_classification"

fine_tune_train_data_dir = './input_data/train_sentences.csv'
fine_tune_test_data_dir = './input_data/valid_sentences.csv'
full_fine_tune_train_data_dir = './input_data/train_text.csv'
full_fine_tune_test_data_dir = './input_data/valid_text.csv'
classify_train_data_dir = './input_data/train_classify_sentences.csv'
classify_test_data_dir = './input_data/valid_classify_sentences.csv'
classify_token_train_data_dir = './input_data/train_classify_tokens.csv'
classify_token_test_data_dir = './input_data/valid_classify_tokens.csv'

fine_tune_model = './fine-tune-models/'
saved_model = 'model_mask'
mtl_model_path = './mtl-models/model_'

label_data_dir = './label_data/'
task_specific_data_dir = './input_data/'
task_specific_model_path = './task-models/'

embedding_model_path = './embedding_models/'
data_augmentation_path = './augmented_data/'
