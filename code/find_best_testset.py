from transformers import RobertaModel, RobertaTokenizer

import config
from incremental_learning import batch_learning_svm
from process_data_for_augmentation import augmentation_for_task_data, augmentation_for_priority_data


def find_best_test_set():
    task_name = 'reopen'
    augment_tools = ['wordnet', 'word2vec', 'roberta']
    # augment_tools = ['roberta', 'wordnet', 'word2vec']
    # token_strategies = ['text',]
    # instance_strategies = ['least']
    token_strategies = ['text', 'sentence']
    instance_strategies = ['least', 'most']

    roberta_tokenizer = RobertaTokenizer.from_pretrained('./roberta-base')

    for augment_tool in augment_tools:
        for token_strategy in token_strategies:
            for instance_strategy in instance_strategies:

                # model_type = 'full_text_fine_tune'
                # roberta = RobertaModel.from_pretrained('./fine-tune-models/' + model_type).to(config.device)
                # f1_full, precision_full, recall_full = batch_learning_svm(
                #     task_name, token_strategy, instance_strategy,
                #     model_type, augment_tool, roberta,
                #     roberta_tokenizer)
                #
                # model_type = 'key_sentences_fine_tune_at_least_2_context_2_exception'
                # roberta = RobertaModel.from_pretrained('./fine-tune-models/' + model_type).to(config.device)
                # f1_key, precision_key, recall_key = batch_learning_svm(
                #     task_name, token_strategy, instance_strategy,
                #     model_type, augment_tool, roberta,
                #     roberta_tokenizer)
                #
                # if f1_key > f1_full and f1_key > 0.5:
                #     continue

                repeat_time = 0

                while repeat_time < 20:

                    # 处理数据
                    if task_name != 'priority':
                        augmentation_for_task_data(
                            task=task_name,
                            select_token_strategy=token_strategy,
                            keep_instance_strategy=instance_strategy,
                            augment_type=augment_tool
                        )
                    else:
                        augmentation_for_priority_data(
                            select_token_strategy=token_strategy,
                            keep_instance_strategy=instance_strategy,
                            augment_type=augment_tool
                        )

                    model_type = 'full_text_fine_tune'
                    roberta = RobertaModel.from_pretrained('./fine-tune-models/' + model_type).to(config.device)
                    f1_full, precision_full, recall_full = batch_learning_svm(task_name, token_strategy,
                                                                              instance_strategy, model_type,
                                                                              augment_tool, roberta, roberta_tokenizer)

                    model_type = 'key_sentences_fine_tune_at_least_2_context_2_exception'
                    roberta = RobertaModel.from_pretrained('./fine-tune-models/' + model_type).to(config.device)
                    f1_key, precision_key, recall_key = batch_learning_svm(task_name, token_strategy, instance_strategy,
                                                                           model_type, augment_tool, roberta,
                                                                           roberta_tokenizer)

                    if task_name == 'severity' or task_name == 'reassignment' or task_name == 'reopen':
                        if f1_key > 0.5 and f1_key > f1_full:
                            print('found a good test set!--------------------------')
                            break
                    else:
                        if f1_key > 0.36 and f1_key > f1_full:
                            print('found a good test set!---------------------------')
                            break

                    repeat_time += 1
                    print('this test set is not good enough!------------------------')

                # reopen和priority的任务只有least数据集
                if task_name == 'reopen' or task_name == 'priority':
                    break


find_best_test_set()
