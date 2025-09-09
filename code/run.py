import os.path

import config
from fine_tune import init_seed, train
from process_data import label_sentences, get_key_sentences, select_key_sentences, split_train_valid_data


def get_model_file_name(sentence_strategy_p, context_strategy_p, context_num_p, k_value_p, remove_strategy_p,
                        stack_trace_strategy_p, fine_tune_strategy_p):
    filename = fine_tune_strategy_p
    filename = filename + ('_at_least_' if sentence_strategy_p != 'top-k' else '_top_') + str(k_value_p)

    if context_strategy_p is False:
        return filename
    filename = filename + '_context_' + str(context_num_p)

    if remove_strategy_p is False:
        return filename
    filename = filename + '_' + stack_trace_strategy_p
    return filename


if __name__ == '__main__':
    fine_tune_strategies = ['prompt tuning']
    sentence_strategies = ['at least k']
    context_strategies = [True, False]
    context_nums = [1, 2, 3]
    k_values = [1, 2, 3]
    remove_strategies = [True, False]
    stack_trace_strategies = ['remove', 'template', 'exception']

    for sentence_strategy in sentence_strategies:
        for context_strategy in context_strategies:
            for context_num in context_nums:
                for k_value in k_values:
                    for remove_strategy in remove_strategies:
                        for stack_trace_strategy in stack_trace_strategies:
                            model_file_name = get_model_file_name(
                                sentence_strategy,
                                context_strategy,
                                context_num,
                                k_value,
                                remove_strategy,
                                stack_trace_strategy,
                                fine_tune_strategies[0]
                            )

                            # 方便断点后继续训练
                            if not os.path.exists('./fine-tune-models/key_sentences_fine_tune_' + model_file_name):
                                # 获取特定关键语句选择策略下，用于模型微调的数据
                                if sentence_strategy == 'at least k':
                                    label_sentences(remove_strategy, stack_trace_strategy)
                                    get_key_sentences(context_strategy, context_num, k_value)
                                elif sentence_strategy == 'top-k':
                                    select_key_sentences(
                                        remove_strategy,
                                        context_strategy,
                                        context_num,
                                        k_value,
                                        stack_trace_strategy
                                    )

                                split_train_valid_data('./label_data/sentences.csv')

                            # 使用特定的微调策略对模型进行微调
                            for fine_tune_strategy in fine_tune_strategies:
                                # 获取微调模型的文件名
                                model_file_name = get_model_file_name(
                                    sentence_strategy,
                                    context_strategy,
                                    context_num,
                                    k_value,
                                    remove_strategy,
                                    stack_trace_strategy,
                                    fine_tune_strategy
                                )
                                if not os.path.exists('./fine-tune-models/key_sentences_fine_tune_' + model_file_name):
                                    init_seed(config.init_seed, config.reproducibility)
                                    train(fine_tune_strategy, model_file_name)
