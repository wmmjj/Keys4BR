"""
在完成数据增强操作的数据处理方法
"""
import collections
import random

import pandas as pd

import config

random.seed(config.init_seed)


# select_token_strategy是指以文本为单位替换词语还是以句子为单位替换词语
# keep_instance_strategy是指保留实例的策略：尽可能多地保留(most)、持平即止(least)
def augmentation_for_task_data(task, select_token_strategy, keep_instance_strategy, augment_type):
    origin_task_file_path = config.task_specific_data_dir + task + ".csv"
    origin_dataset = pd.read_csv(origin_task_file_path)
    # print(len(origin_dataset))

    augmentation_file_path = config.data_augmentation_path + 'full_text_augmented_' + select_token_strategy + '_' + augment_type + '.csv'
    augmentation_dataset = pd.read_csv(augmentation_file_path)

    res_file_path = config.task_specific_data_dir + task + "_" + select_token_strategy + "_data_augmentation_" + keep_instance_strategy + '_' + augment_type + ".csv"
    res_columns = ['project', 'bug_id', 'full_text', task, 'is_origin', 'is_test']

    label_0_origin = []
    label_0_augment = []
    label_1_origin = []
    label_1_augment = []

    # 抽取出一部分真实数据作为测试集
    test_df = origin_dataset.groupby(task).apply(lambda x: x.sample(frac=0.2))
    test_df.reset_index(drop=True, inplace=True)
    test_df = test_df.sample(frac=1, random_state=config.init_seed)
    test_bug_id = test_df['bug_id'].tolist()
    # print(collections.Counter(test_df[task].tolist()))
    # print(test_df)

    test_origin_data_label_0 = []
    test_origin_data_label_1 = []
    test_augment_data_label_0 = []
    test_augment_data_label_1 = []

    for idx, row in origin_dataset.iterrows():
        bug_id = row['bug_id']
        label = row[task]
        # print(bug_id)

        augmentation_data_line = augmentation_dataset.loc[augmentation_dataset['bug_id'] == int(bug_id)]
        if len(augmentation_data_line) == 0:
            augmentation_data_line = augmentation_dataset.loc[augmentation_dataset['bug_id'] == str(bug_id)]

        if int(bug_id) in test_bug_id or str(bug_id) in test_bug_id:
            if label == 1:
                test_origin_data_label_1.append([row['project'], bug_id, row['full_text'], label, 1, 1])
            else:
                test_origin_data_label_0.append([row['project'], bug_id, row['full_text'], label, 1, 1])
            if len(augmentation_data_line) != 0:
                augmentation_texts = str(augmentation_data_line['augmented_text'].values[0]).split("\n")
                if label == 1:
                    for text in augmentation_texts:
                        test_augment_data_label_1.append([row['project'], bug_id, text, label, 0, 1])
                else:
                    for text in augmentation_texts:
                        test_augment_data_label_0.append([row['project'], bug_id, text, label, 0, 1])
            continue

        if len(augmentation_data_line) == 0:
            if label == 1:
                label_1_origin.append([row['project'], bug_id, row['full_text'], label, 1, 0])
            else:
                label_0_origin.append([row['project'], bug_id, row['full_text'], label, 1, 0])
            continue

        augmentation_texts = str(augmentation_data_line['augmented_text'].values[0]).split("\n")
        # 把每个标签下的原始文本与增强文本分别抽取出来
        if label == 1:
            label_1_origin.append([row['project'], bug_id, augmentation_data_line['full_text'].values[0], label, 1, 0])
            for text in augmentation_texts:
                label_1_augment.append([row['project'], bug_id, text, label, 0, 0])
        else:
            label_0_origin.append([row['project'], bug_id, augmentation_data_line['full_text'].values[0], label, 1, 0])
            for text in augmentation_texts:
                label_0_augment.append([row['project'], bug_id, text, label, 0, 0])

    random.shuffle(label_0_augment)
    random.shuffle(label_1_augment)

    # augment_data = augmentation_dataset[augmentation_dataset['bug_id'] not in test_bug_id]

    # print('size of label_0_origin: {}; size of label_0_augment: {}'.format(len(label_0_origin), len(label_0_augment)))
    # print('size of label_1_origin: {}; size of label_1_augment: {}'.format(len(label_1_origin), len(label_1_augment)))
    if keep_instance_strategy == 'most':
        if len(label_0_origin) > len(label_1_origin):
            # 少数类扩展至最大
            label_1_origin.extend(label_1_augment)
            # 计算多数类需要采样的数量
            sample_count = len(label_1_origin) - len(label_0_origin)
            # 少数类扩充后仍然比多数类少
            if sample_count < 0:
                # 减少多数类的数量
                label_0_origin = random.sample(label_0_origin, len(label_1_origin))
            else:
                # 随机采样扩展的样本
                label_0_origin.extend(random.sample(label_0_augment, sample_count))
        else:
            label_0_origin.extend(label_0_augment)
            sample_count = len(label_0_origin) - len(label_1_origin)
            if sample_count < 0:
                label_1_origin = random.sample(label_1_origin, len(label_0_origin))
            else:
                label_1_origin.extend(random.sample(label_1_augment, sample_count))
    else:
        if len(label_0_origin) > len(label_1_origin):
            # 根据少数类和多数类的数量差距计算采样数量
            sample_count = len(label_0_origin) - len(label_1_origin)
            # if sample_count > 0:
            # 扩展的样本数量大于需要采样的数量
            if sample_count < len(label_1_augment):
                # 直接从扩展样本中进行采样
                label_1_origin.extend(random.sample(label_1_augment, sample_count))
            else:
                # 少数类扩展到最大规模
                label_1_origin.extend(label_1_augment)
                # 多数类规模缩小为少数类的数量
                label_0_origin = random.sample(label_0_origin, len(label_1_origin))
        # else:
        # 	label_1_origin = random.sample(label_1_origin, len(label_0_origin))
        else:
            sample_count = len(label_1_origin) - len(label_0_origin)
            # if sample_count > 0:
            if sample_count < len(label_0_augment):
                label_0_origin.extend(random.sample(label_0_augment, sample_count))
            else:
                label_0_origin.extend(label_0_augment)
                label_1_origin = random.sample(label_1_origin, len(label_0_origin))
        # else:
        # 	label_0_origin = random.sample(label_0_origin, len(label_1_origin))

    # 训练集
    label_0_origin.extend(label_1_origin)
    random.shuffle(label_0_origin)

    # 添加测试集，根据is_test值区分
    # 先将origin的数据进行处理
    # if len(test_origin_data_label_0) == len(test_origin_data_label_1):
    # 	test_origin_data_label_0.extend(test_origin_data_label_1)
    # # 某个类别的原始测试样例比较少，则先从augment的测试样例中采样填补，若仍不足，则减少多数类的测试样例数量
    # elif len(test_origin_data_label_0) > len(test_origin_data_label_1):
    # 	sample_count = len(test_origin_data_label_0) - len(test_origin_data_label_1)
    # 	if sample_count > len(test_augment_data_label_1):
    # 		test_origin_data_label_0 = random.sample(test_origin_data_label_0, (len(test_origin_data_label_0) - (sample_count - len(test_augment_data_label_1))))
    # 		test_origin_data_label_0.extend(test_augment_data_label_1)
    # 	else:
    # 		test_origin_data_label_0.extend(random.sample(test_augment_data_label_1, sample_count))
    # 	test_origin_data_label_0.extend(test_origin_data_label_1)
    # else:
    # 	sample_count = len(test_origin_data_label_1) - len(test_origin_data_label_0)
    # 	if sample_count > len(test_origin_data_label_0):
    # 		test_origin_data_label_1 = random.sample(test_origin_data_label_1, (len(test_origin_data_label_1) - (sample_count - len(test_augment_data_label_0))))
    # 		test_origin_data_label_0.extend(test_augment_data_label_0)
    # 	else:
    # 		test_origin_data_label_0.extend(random.sample(test_augment_data_label_0, sample_count))
    # 	test_origin_data_label_0.extend(test_origin_data_label_1)
    # # 添加相同数量的标签数据
    # if len(test_augment_data_label_1) == len(test_augment_data_label_0):
    # 	test_origin_data_label_0.extend(test_augment_data_label_1)
    # 	test_origin_data_label_0.extend(test_augment_data_label_0)
    # elif len(test_augment_data_label_1) > len(test_augment_data_label_0):
    # 	test_origin_data_label_0.extend(test_augment_data_label_0)
    # 	test_origin_data_label_0.extend(random.sample(test_augment_data_label_1, len(test_augment_data_label_0)))
    # else:
    # 	test_origin_data_label_0.extend(test_augment_data_label_1)
    # 	test_origin_data_label_0.extend(random.sample(test_augment_data_label_0, len(test_augment_data_label_1)))
    test_origin_data_label_0.extend(test_origin_data_label_1)
    random.shuffle(test_origin_data_label_0)
    label_0_origin.extend(test_origin_data_label_0)

    res_data = pd.DataFrame(label_0_origin, columns=res_columns)
    res_data.to_csv(res_file_path, index=False)


def augment_more_evenly(origin_instance_count, sample_count, augment_data):
    if sample_count < origin_instance_count:
        return random.sample(augment_data, sample_count)

    augment_data = augment_data.dropna(subset=['augmented_text'])

    instance_map = {}
    # 采样数量大于原始的样本数量，则每个bug报告的扩展样本都随机抽取一个
    while sample_count > origin_instance_count:
        for idx, row in augment_data.iterrows():
            augment_instances = str(row['augmented_text']).split('\n')
            instance = sample_single_bug_data(augment_instances, instance_map)
            if instance != "":
                instance_map[instance] = 1
                sample_count -= 1
    # 采样数量不足原始样本数量，则随机采样
    while sample_count > 0:
        # 采样一部分数据
        sampled_data = augment_data.sample(sample_count)

        for idx, row in sampled_data.iterrows():
            augment_instances = str(row['augmented_text']).split('\n')
            instance = sample_single_bug_data(augment_instances, instance_map)
            if instance != "":
                instance_map[instance] = 1
                sample_count -= 1
        # 避免有些数据已经被采样完
        if sample_count > 0:
            sampled_id = sampled_data['bug_id'].tolist()
            augment_data = augment_data[augment_data['bug_id'] not in sampled_id]


def sample_single_bug_data(augment_instances, instance_map):
    if len(augment_instances) > 0:
        instance = random.sample(augment_instances, 1)[0]
        retry_times = len(augment_instances) - 1
        try_times = 0
        while instance in instance_map and try_times < retry_times:
            augment_instances.remove(instance)
            instance = random.sample(augment_instances, 1)[0]
            try_times += 1
        if try_times < retry_times:
            return instance
        else:
            return ""


def augmentation_for_priority_data(select_token_strategy, keep_instance_strategy, augment_type):
    origin_task_file_path = config.task_specific_data_dir + "priority.csv"
    origin_dataset = pd.read_csv(origin_task_file_path)

    augmentation_file_path = config.data_augmentation_path + 'full_text_augmented_' + select_token_strategy + '_' + augment_type + '.csv'
    augmentation_dataset = pd.read_csv(augmentation_file_path)

    res_file_path = config.task_specific_data_dir + "priority_" + select_token_strategy + "_data_augmentation_" + keep_instance_strategy + '_' + augment_type + ".csv"
    # res_columns = ['project', 'bug_id', 'full_text', 'priority', 'is_origin']
    res_columns = ['project', 'bug_id', 'full_text', 'priority', 'is_origin', 'is_test']

    # 抽取出一部分真实数据作为测试集
    test_df = origin_dataset.groupby('priority').apply(lambda x: x.sample(frac=0.2))
    test_df.reset_index(drop=True, inplace=True)
    test_df = test_df.sample(frac=1, random_state=config.init_seed)
    test_bug_id = test_df['bug_id'].tolist()

    label_0_origin = []
    label_0_augment = []
    label_1_origin = []
    label_1_augment = []
    label_2_origin = []
    label_2_augment = []

    test_origin_data_label_0 = []
    test_origin_data_label_1 = []
    test_origin_data_label_2 = []
    test_augment_data_label_0 = []
    test_augment_data_label_1 = []
    test_augment_data_label_2 = []

    for idx, row in origin_dataset.iterrows():
        bug_id = row['bug_id']
        label = row['priority']
        # print(bug_id)

        augmentation_data_line = augmentation_dataset.loc[augmentation_dataset['bug_id'] == int(bug_id)]
        if len(augmentation_data_line) == 0:
            augmentation_data_line = augmentation_dataset.loc[augmentation_dataset['bug_id'] == str(bug_id)]

        if int(bug_id) in test_bug_id or str(bug_id) in test_bug_id:
            if label == 0:
                test_origin_data_label_0.append([row['project'], bug_id, row['full_text'], label, 1, 1])
            elif label == 1:
                test_origin_data_label_1.append([row['project'], bug_id, row['full_text'], label, 1, 1])
            else:
                test_origin_data_label_2.append([row['project'], bug_id, row['full_text'], label, 1, 1])
            if len(augmentation_data_line) != 0:
                augmentation_texts = str(augmentation_data_line['augmented_text'].values[0]).split("\n")
                if label == 1:
                    for text in augmentation_texts:
                        test_augment_data_label_1.append([row['project'], bug_id, text, label, 0, 1])
                elif label == 0:
                    for text in augmentation_texts:
                        test_augment_data_label_0.append([row['project'], bug_id, text, label, 0, 1])
                else:
                    for text in augmentation_texts:
                        test_augment_data_label_2.append([row['project'], bug_id, text, label, 0, 1])
            continue

        if len(augmentation_data_line) == 0:
            if label == 1:
                label_1_origin.append([row['project'], bug_id, row['full_text'], label, 1, 0])
            elif label == 0:
                label_0_origin.append([row['project'], bug_id, row['full_text'], label, 1, 0])
            else:
                label_2_origin.append([row['project'], bug_id, row['full_text'], label, 1, 0])
            continue

        augmentation_texts = str(augmentation_data_line['augmented_text'].values[0]).split("\n")
        # 把每个标签下的原始文本与增强文本分别抽取出来
        if label == 1:
            label_1_origin.append([row['project'], bug_id, augmentation_data_line['full_text'].values[0], label, 1, 0])
            for text in augmentation_texts:
                label_1_augment.append([row['project'], bug_id, text, label, 0, 0])
        elif label == 0:
            label_0_origin.append([row['project'], bug_id, augmentation_data_line['full_text'].values[0], label, 1, 0])
            for text in augmentation_texts:
                label_0_augment.append([row['project'], bug_id, text, label, 0, 0])
        else:
            label_2_origin.append([row['project'], bug_id, augmentation_data_line['full_text'].values[0], label, 1, 0])
            for text in augmentation_texts:
                label_2_augment.append([row['project'], bug_id, text, label, 0, 0])

    random.shuffle(label_0_augment)
    random.shuffle(label_1_augment)
    random.shuffle(label_2_augment)

    # print('size of label_0_origin: {}; size of label_0_augment: {}'.format(len(label_0_origin), len(label_0_augment)))
    # print('size of label_1_origin: {}; size of label_1_augment: {}'.format(len(label_1_origin), len(label_1_augment)))
    # print('size of label_2_origin: {}; size of label_2_augment: {}'.format(len(label_2_origin), len(label_2_augment)))

    if len(label_0_origin) <= len(label_1_origin) and len(label_0_origin) <= len(label_2_origin):
        label_least_origin = label_0_origin
        label_least_augment = label_0_augment
        if len(label_1_origin) <= len(label_2_origin):
            label_mid_origin = label_1_origin
            label_mid_augment = label_1_augment
            label_most_origin = label_2_origin
            label_most_augment = label_2_augment
        else:
            label_mid_origin = label_2_origin
            label_mid_augment = label_2_augment
            label_most_origin = label_1_origin
            label_most_augment = label_1_augment
    elif len(label_1_origin) <= len(label_0_origin) and len(label_1_origin) <= len(label_2_origin):
        label_least_origin = label_1_origin
        label_least_augment = label_1_augment
        if len(label_0_origin) <= len(label_2_origin):
            label_mid_origin = label_0_origin
            label_mid_augment = label_0_augment
            label_most_origin = label_2_origin
            label_most_augment = label_2_augment
        else:
            label_mid_origin = label_2_origin
            label_mid_augment = label_2_augment
            label_most_origin = label_0_origin
            label_most_augment = label_0_augment
    else:
        label_least_origin = label_2_origin
        label_least_augment = label_2_augment
        if len(label_0_origin) <= len(label_1_origin):
            label_mid_origin = label_0_origin
            label_mid_augment = label_0_augment
            label_most_origin = label_1_origin
            label_most_augment = label_1_augment
        else:
            label_mid_origin = label_1_origin
            label_mid_augment = label_1_augment
            label_most_origin = label_0_origin
            label_most_augment = label_0_augment

    # 数据量最少的类别使用所有的数据
    label_least_origin.extend(label_least_augment)

    # origin数据量少于label_least_origin数据量，则从augment中采样
    if len(label_mid_origin) < len(label_least_origin):
        label_mid_origin.extend(random.sample(label_mid_augment, (len(label_least_origin) - len(label_mid_origin))))
    # 否则从origin中采样
    else:
        label_mid_origin = random.sample(label_mid_origin, len(label_least_origin))

    if len(label_most_origin) < len(label_least_origin):
        label_most_origin.extend(random.sample(label_most_augment, (len(label_least_origin) - len(label_most_origin))))
    else:
        label_most_origin = random.sample(label_most_origin, len(label_least_origin))

    # 三个类别的数据混合并打乱顺序
    label_least_origin.extend(label_mid_origin)
    label_least_origin.extend(label_most_origin)
    random.shuffle(label_least_origin)

    # 添加相同数量的标签数据
    # if len(test_origin_data_label_0) == len(test_origin_data_label_1) == len(test_origin_data_label_2):
    # 	test_origin_data_label_0.extend(test_origin_data_label_1)
    # 	test_origin_data_label_0.extend(test_origin_data_label_2)
    # else:
    # 	if len(test_origin_data_label_1) >= len(test_origin_data_label_0) and len(test_origin_data_label_2) >= len(test_origin_data_label_0):
    # 		test_min_origin = test_origin_data_label_0
    # 		test_min_augment = test_augment_data_label_0
    # 		if len(test_origin_data_label_1) >= len(test_origin_data_label_2):
    # 			test_mid_origin	= test_origin_data_label_1
    # 			test_mid_augment = test_augment_data_label_1
    # 			test_max_origin = test_origin_data_label_2
    # 			test_max_augment = test_augment_data_label_2
    # 		else:
    # 			test_mid_origin = test_origin_data_label_2
    # 			test_mid_augment = test_augment_data_label_2
    # 			test_max_origin = test_origin_data_label_1
    # 			test_max_augment = test_augment_data_label_1
    # 	elif len(test_origin_data_label_0) >= len(test_origin_data_label_1) and len(test_origin_data_label_2) >= len(test_origin_data_label_1):
    # 		test_min_origin = test_origin_data_label_1
    # 		test_min_augment = test_augment_data_label_1
    # 		if len(test_origin_data_label_0) >= len(test_origin_data_label_2):
    # 			test_mid_origin	= test_origin_data_label_0
    # 			test_mid_augment = test_augment_data_label_0
    # 			test_max_origin = test_origin_data_label_2
    # 			test_max_augment = test_augment_data_label_2
    # 		else:
    # 			test_mid_origin = test_origin_data_label_2
    # 			test_mid_augment = test_augment_data_label_2
    # 			test_max_origin = test_origin_data_label_0
    # 			test_max_augment = test_augment_data_label_0
    # 	else:
    # 		test_min_origin = test_origin_data_label_2
    # 		test_min_augment = test_augment_data_label_2
    # 		if len(test_origin_data_label_1) >= len(test_origin_data_label_0):
    # 			test_mid_origin	= test_origin_data_label_1
    # 			test_mid_augment = test_augment_data_label_1
    # 			test_max_origin = test_origin_data_label_0
    # 			test_max_augment = test_augment_data_label_0
    # 		else:
    # 			test_mid_origin = test_origin_data_label_0
    # 			test_mid_augment = test_augment_data_label_0
    # 			test_max_origin = test_origin_data_label_1
    # 			test_max_augment = test_augment_data_label_1
    #
    # 	sample_count = len(test_max_origin) - len(test_min_origin)
    # 	if sample_count > len(test_min_augment):
    # 		test_min_origin.extend(test_min_augment)
    # 		test_max_origin = random.sample(test_max_origin, len(test_min_origin))
    # 	else:
    # 		test_min_origin.extend(random.sample(test_min_augment, sample_count))
    #
    # 	if len(test_min_origin) > len(test_mid_origin):
    # 		test_mid_origin.extend(random.sample(test_mid_augment, len(test_min_origin) - len(test_mid_origin)))
    # 	else:
    # 		test_mid_origin = random.sample(test_mid_origin, len(test_min_origin))
    #
    # 	assert len(test_min_origin) == len(test_mid_origin) == len(test_max_origin)
    # 	test_min_origin.extend(test_mid_origin)
    # 	test_min_origin.extend(test_max_origin)
    # 	test_origin_data_label_0 = test_min_origin

    test_origin_data_label_0.extend(test_origin_data_label_1)
    test_origin_data_label_0.extend(test_origin_data_label_2)
    random.shuffle(test_origin_data_label_0)
    label_least_origin.extend(test_origin_data_label_0)

    res = pd.DataFrame(label_least_origin, columns=res_columns)
    res.to_csv(res_file_path, index=False)


def split_augmentation_train_test_data(task, class_num, select_token_strategy, keep_instance_strategy):
    total_data_file = config.task_specific_data_dir + task + "_" + select_token_strategy + "_data_augmentation_" + keep_instance_strategy + ".csv"
    total_dataset = pd.read_csv(total_data_file)

    res_columns = total_dataset.columns
    res_file_path_prefix = config.task_specific_data_dir + task + "_" + select_token_strategy + "_data_augmentation_" + keep_instance_strategy

    train_set = []
    test_set = []

    for idx in range(class_num):
        corr_bug_list = total_dataset[total_dataset[task] == idx].values.tolist()
        random.shuffle(corr_bug_list)
        train_set_size = int(0.8 * len(corr_bug_list))
        train_set += corr_bug_list[: train_set_size]
        test_set += corr_bug_list[train_set_size:]

    random.shuffle(train_set)
    random.shuffle(test_set)

    res_data = pd.DataFrame(train_set, columns=res_columns)
    res_data.to_csv(res_file_path_prefix + "_train.csv", index=False)
    res_data = pd.DataFrame(test_set, columns=res_columns)
    res_data.to_csv(res_file_path_prefix + "_test.csv", index=False)


if __name__ == "__main__":
    task_name = "reassignment"
    # category_size = 2
    augment_token_selection_strategy = "text"
    instance_augmentation_strategy = "least"
    augment_tool_type = "roberta"
    if task_name != 'priority':
        augmentation_for_task_data(
            task=task_name,
            select_token_strategy=augment_token_selection_strategy,
            keep_instance_strategy=instance_augmentation_strategy,
            augment_type=augment_tool_type
        )
    else:
        augmentation_for_priority_data(
            select_token_strategy=augment_token_selection_strategy,
            keep_instance_strategy=instance_augmentation_strategy,
            augment_type=augment_tool_type
        )

# split_augmentation_train_test_data(
# 	task=task_name,
# 	class_num=category_size,
# 	select_token_strategy=augment_token_selection_strategy,
# 	keep_instance_strategy=instance_augmentation_strategy
# )
