import collections
import os.path
import pickle
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from transformers import RobertaModel, RobertaTokenizer
import pandas as pd
import config


def batch_learning_svm(task, augment_token_selection_strategy, instance_augmentation_strategy, fine_tune_model,
                       augment_type, roberta_model, tokenizer):
    model = SGDClassifier(loss='hinge', max_iter=100, random_state=config.init_seed)

    batch_size = 1000
    res_filename_body = task + "_" + augment_token_selection_strategy + '_data_augmentation_' + instance_augmentation_strategy + '_' + fine_tune_model + '_' + augment_type
    data_filename_body = task + "_" + augment_token_selection_strategy + '_data_augmentation_' + instance_augmentation_strategy + '_' + augment_type
    augment_data_file = config.task_specific_data_dir + data_filename_body + ".csv"
    print(res_filename_body)

    # 获取特征表示和标签
    features_file_path = './roberta_features/' + res_filename_body

    # if instance_augmentation_strategy == 'least':
    #     if os.path.exists(features_file_path + '.pkl'):
    #         # print('found the saved features')
    #         features, labels = get_features_from_file(features_file_path)
    #     else:
    #         # print('cannot find the saved features, compute them right now')
    #         features, labels = compute_features(task, features_file_path, augment_data_file, fine_tune_model)
    #
    #     assert len(features) == len(labels)
    #
    #     # 获取训练集
    #     # train_size = int(len(features) * 0.8)
    #     # x_train = features[: train_size]
    #     # y_train = labels[: train_size]
    #
    #     # for epoch in range(config.epochs):
    #     #     print("epoch number: {}".format(epoch))
    #     if task != 'priority':
    #         for batch_features, batch_labels in batch_generator(x_train, y_train, batch_size):
    #             model.partial_fit(batch_features, batch_labels, classes=[0, 1])
    #     else:
    #         for batch_features, batch_labels in batch_generator(x_train, y_train, batch_size):
    #             model.partial_fit(batch_features, batch_labels, classes=[0, 1, 2])
    #
    #     x_test = features[train_size:]
    #     y_test = labels[train_size:]
    #     # print('predicting...')
    #     y_pred = model.predict(x_test)
    #
    # else:
    # dataset = pd.read_csv(augment_data_file, usecols=['full_text', task])
    dataset = pd.read_csv(augment_data_file, usecols=['full_text', task, 'is_test'])
    # print(dataset.columns)

    train_dataset = dataset.loc[dataset['is_test'] == 0]
    test_dataset = dataset.loc[dataset['is_test'] == 1]

    # text_list = dataset['full_text'].tolist()
    # label_list = dataset[task].tolist()
    #
    # train_size = int(len(label_list) * 0.8)
    # x_train = text_list[: train_size]
    # y_train = label_list[: train_size]

    x_train = train_dataset['full_text'].tolist()
    y_train = train_dataset[task].tolist()

    # model_file = './fine-tune-models/' + fine_tune_model
    # roberta_model = RobertaModel.from_pretrained(model_file).to(config.device)
    # tokenizer = RobertaTokenizer.from_pretrained('./roberta-base')

    # for epoch in range(config.epochs):
    #     print("epoch number: {}".format(epoch))
    if task != 'priority':
        for batch_features, batch_labels in batch_generator_get_features(x_train, y_train, batch_size, roberta_model,
                                                                         tokenizer, features_file_path):
            model.partial_fit(batch_features, batch_labels, classes=[0, 1])
    else:
        for batch_features, batch_labels in batch_generator_get_features(x_train, y_train, batch_size, roberta_model,
                                                                         tokenizer, features_file_path):
            model.partial_fit(batch_features, batch_labels, classes=[0, 1, 2])

    # x_test = text_list[train_size:]
    # x_test = get_features_for_batch(x_test, roberta_model, tokenizer)
    # y_test = label_list[train_size:]
    # print('predicting...')

    x_test = test_dataset['full_text'].tolist()
    x_test = get_features_for_batch(x_test, roberta_model, tokenizer)
    y_test = test_dataset[task].tolist()
    y_pred = model.predict(x_test)

    print('saving model...')
    # model_save_path = './svm_model/' + res_filename_body + '.pkl'
    # joblib.dump(model, model_save_path)
    if task != 'priority':
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
    else:
        f1 = f1_score(y_test, y_pred, average='micro')
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='micro')
        recall = recall_score(y_test, y_pred, average='micro')

    print(collections.Counter(y_test))
    print("predict_res: {}".format(y_pred))
    print(collections.Counter(y_pred))
    print("F-score: {}".format(f1))
    print("Accuracy-score: {}".format(acc))
    print("Precision-score: {}".format(precision))
    print("Recall-score: {}".format(recall))


# 从pickle文件中加载特征
def get_features_from_file(features_file_path):
    # features_file_path = './roberta_features/' + task + '_' + augment_token_selection_strategy + '_data_augmentation_' + instance_augmentation_strategy + '.pkl'
    with open(features_file_path + '.pkl', 'rb') as f:
        features = pickle.load(f)
    with open(features_file_path + '_label.pkl', 'rb') as f:
        labels = pickle.load(f)

    return features, labels


# 使用Roberta模型获取文本编码
def compute_features(task, features_path, augment_data_file, roberta_model, tokenizer):
    # model_file = './fine-tune-models/' + fine_tune_model
    # roberta_model = RobertaModel.from_pretrained(model_file).to(config.device)
    # tokenizer = RobertaTokenizer.from_pretrained('./roberta-base')

    # augment_data_file_prefix = task + "_" + augment_token_selection_strategy + "_data_augmentation_" + instance_augmentation_strategy
    dataset = pd.read_csv(augment_data_file, usecols=['full_text', task])
    text_list = dataset['full_text'].tolist()
    label_list = dataset[task].tolist()

    features = get_features_for_batch(text_list, roberta_model, tokenizer)

    # 将编码后的数据保存至文件中
    # with open(features_path + '.pkl', 'wb') as f:
    #     pickle.dump(features, f)
    #
    # with open(features_path + '_label.pkl', 'wb') as f:
    #     pickle.dump(label_list, f)

    return features, label_list


def batch_generator(features, labels, batch_size):
    num_samples = len(features)
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)
        batch_features = features[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        yield batch_features, batch_labels


def batch_generator_get_features(texts, labels, batch_size, roberta_model, tokenizer, features_path):
    num_samples = len(labels)
    num_batches = (num_samples + batch_size - 1) // batch_size
    # print("batch numbers: {}".format(num_batches))

    for batch_idx in range(num_batches):
        # 判断是否有保存的文件
        # 有则直接加载文件
        if os.path.exists(features_path + '_' + str(batch_idx) + '.pkl'):
            with open(features_path + '_' + str(batch_idx) + '.pkl', 'rb') as f:
                # print('getting batch features from existing file: {}'.format(batch_idx))
                batch_features = pickle.load(f)
            with open(features_path + '_' + str(batch_idx) + '_label', 'rb') as f:
                # print('getting batch labels...')
                batch_labels = pickle.load(f)
        # 没有则现场生成
        else:
            # print("generate batch: {}".format(batch_idx))
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_text = texts[start_idx:end_idx]
            batch_features = get_features_for_batch(batch_text, roberta_model, tokenizer)
            batch_labels = labels[start_idx:end_idx]
            # with open(features_path + '_' + str(batch_idx) + '.pkl', 'wb') as f:
            #     print('saving batch features: {}'.format(batch_idx))
            #     pickle.dump(batch_features, f)
            # with open(features_path + '_' + str(batch_idx) + '_label', 'wb') as f:
            #     print('saving batch labels...')
            #     pickle.dump(batch_labels, f)

        yield batch_features, batch_labels


def get_features_for_batch(text_list, roberta_model, tokenizer):
    features = []
    # print('getting features...')
    for text in text_list:
        text = tokenizer(str(text), return_tensors='pt', truncation=True, max_length=512)
        text = {key: value.to(config.device) for key, value in text.items()}
        output = roberta_model(**text)
        cls = output[0][:, 0, :][0].detach().cpu().numpy()
        features.append(list(cls))

    return features


if __name__ == '__main__':
    token_strategies = ['sentence']
    instance_strategies = ['most']
    # token_strategies = ['text', 'sentence']
    # instance_strategies = ['most', 'least']
    # task_names = ['reassignment']
    task_names = ['severity']
    # task_names = ['priority', 'reopen']
    # model_types = ['full_text_fine_tune', 'key_sentences_fine_tune_at_least_3_context_1_template']
    # model_types = [
    #     'full_text_fine_tune',
    #     'key_sentences_fine_tune_at_least_2_context_exception',
    #     'key_sentences_fine_tune_top_2_context_exception',
    #     'key_sentences_fine_tune_top_2_context_template',
    #     'key_sentences_fine_tune_top_2_context_remove',
    #     'key_sentences_fine_tune_top_2_context',
    #     'key_sentences_fine_tune_top_2',
    #     'key_sentences_fine_tune_at_least_2_context_template',
    #     'key_sentences_fine_tune_at_least_2_context_remove',
    #     'key_sentences_fine_tune_at_least_2_context',
    #     'key_sentences_fine_tune_at_least_2',
    # ]
    model_types = [
        # 'full_text_fine_tune',
        'key_sentences_fine_tune_top_3_context_1_exception',
        'key_sentences_fine_tune_top_3_context_1_template',
        'key_sentences_fine_tune_top_3_context_1_remove',
        'key_sentences_fine_tune_top_3_context_1',
        'key_sentences_fine_tune_top_3',
        'key_sentences_fine_tune_at_least_3_context_1_exception',
        'key_sentences_fine_tune_at_least_3_context_1_template',
        'key_sentences_fine_tune_at_least_3_context_1_remove',
        'key_sentences_fine_tune_at_least_3_context_1',
        # 'key_sentences_fine_tune_at_least_3',
    ]
    # model_types = ['key_sentences_fine_tune_top_3', 'key_sentences_fine_tune_top_3_context_1']
    # model_types = ['key_sentences_fine_tune_at_least_2_context', 'key_sentences_fine_tune_at_least_2_template']
    # model_types = ['key_sentences_fine_tune_top_2_context_template', 'full_text_fine_tune', 'key_sentences_fine_tune']
    augment_tool_types = ['wordnet']
    # augment_tool_types = ['wordnet', 'word2vec', 'roberta']
    for model_type in model_types:
        roberta = RobertaModel.from_pretrained('./fine-tune-models/' + model_type).to(config.device)
        roberta_tokenizer = RobertaTokenizer.from_pretrained('./roberta-base')
        for augment_tool_type in augment_tool_types:
            for task_name in task_names:
                if task_name == 'reopen' or task_name == 'priority':
                    batch_learning_svm(task_name, 'text', 'least', model_type, augment_tool_type, roberta,
                                       roberta_tokenizer)
                    batch_learning_svm(task_name, 'sentence', 'least', model_type, augment_tool_type, roberta,
                                       roberta_tokenizer)
                # batch_learning_svm(task_name, 'text', 'least', 'full_text_fine_tune', augment_tool_type)
                # batch_learning_svm(task_name, 'sentence', 'least', 'full_text_fine_tune', augment_tool_type)
                else:
                    for token_strategy in token_strategies:
                        for instance_strategy in instance_strategies:
                            batch_learning_svm(task_name, token_strategy, instance_strategy, model_type,
                                               augment_tool_type, roberta, roberta_tokenizer)
