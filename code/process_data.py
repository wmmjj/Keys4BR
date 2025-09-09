import math
import random
import re

import numpy as np
import pandas as pd
from xml.dom.minidom import parse
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from sklearn.utils import shuffle

import config

doc_name = ["AspectJ", "Birt", "Eclipse_Platform_UI", "JDT", "SWT"]

EngStopWord = ["a", "a's", "able", "about", "above",
               "according", "accordingly", "across", "actually", "after",
               "afterwards", "again", "against", "ain't", "all", "allow",
               "allows", "almost", "alone", "along", "already", "also",
               "although", "always", "am", "among", "amongst", "an", "and",
               "another", "any", "anybody", "anyhow", "anyone", "anything",
               "anyway", "anyways", "anywhere", "apart", "appear",
               "appreciate", "appropriate", "are", "aren't", "around", "as",
               "aside", "ask", "asking", "associated", "at", "available",
               "away", "awfully", "b", "be", "became", "because", "become",
               "becomes", "becoming", "been", "before", "beforehand",
               "behind", "being", "believe", "below", "beside", "besides",
               "best", "better", "between", "beyond", "both", "brief", "but",
               "by", "c", "c'mon", "c's", "came", "can", "can't", "cannot",
               "cant", "cause", "causes", "certain", "certainly", "changes",
               "clearly", "co", "com", "come", "comes", "concerning",
               "consequently", "consider", "considering", "contain",
               "containing", "contains", "corresponding", "could", "couldn't",
               "course", "currently", "d", "definitely", "described",
               "despite", "did", "didn't", "different", "do", "does",
               "doesn't", "doing", "don't", "done", "down", "downwards",
               "during", "e", "each", "edu", "eg", "eight", "either", "else",
               "elsewhere", "enough", "entirely", "especially", "et", "etc",
               "even", "ever", "every", "everybody", "everyone", "everything",
               "everywhere", "ex", "exactly", "example", "except", "f", "far",
               "few", "fifth", "first", "five", "followed", "following",
               "follows", "for", "former", "formerly", "forth", "four",
               "from", "further", "furthermore", "g", "get", "gets",
               "getting", "given", "gives", "go", "goes", "going", "gone",
               "got", "gotten", "greetings", "h", "had", "hadn't", "happens",
               "hardly", "has", "hasn't", "have", "haven't", "having", "he",
               "he's", "hello", "help", "hence", "her", "here", "here's",
               "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
               "hi", "him", "himself", "his", "hither", "hopefully", "how",
               "howbeit", "however", "i", "i'd", "i'll", "i'm", "i've", "ie",
               "if", "ignored", "immediate", "in", "inasmuch", "inc",
               "indeed", "indicate", "indicated", "indicates", "inner",
               "insofar", "instead", "into", "inward", "is", "isn't", "it",
               "it'd", "it'll", "it's", "its", "itself", "j", "just", "k",
               "keep", "keeps", "kept", "know", "knows", "known", "l", "last",
               "lately", "later", "latter", "latterly", "least", "less",
               "lest", "let", "let's", "like", "liked", "likely", "little",
               "look", "looking", "looks", "ltd", "m", "mainly", "many",
               "may", "maybe", "me", "mean", "meanwhile", "merely", "might",
               "more", "moreover", "most", "mostly", "much", "must", "my",
               "myself", "n", "name", "namely", "nd", "near", "nearly",
               "necessary", "need", "needs", "neither", "never",
               "nevertheless", "new", "next", "nine", "no", "nobody", "non",
               "none", "noone", "nor", "normally", "not", "nothing", "novel",
               "now", "nowhere", "o", "obviously", "of", "off", "often", "oh",
               "ok", "okay", "old", "on", "once", "one", "ones", "only",
               "onto", "or", "other", "others", "otherwise", "ought", "our",
               "ours", "ourselves", "out", "outside", "over", "overall",
               "own", "p", "particular", "particularly", "per", "perhaps",
               "placed", "please", "plus", "possible", "presumably",
               "probably", "provides", "q", "que", "quite", "qv", "r",
               "rather", "rd", "re", "really", "reasonably", "regarding",
               "regardless", "regards", "relatively", "respectively", "right",
               "s", "said", "same", "saw", "say", "saying", "says", "second",
               "secondly", "see", "seeing", "seem", "seemed", "seeming",
               "seems", "seen", "self", "selves", "sensible", "sent",
               "serious", "seriously", "seven", "several", "shall", "she",
               "should", "shouldn't", "since", "six", "so", "some",
               "somebody", "somehow", "someone", "something", "sometime",
               "sometimes", "somewhat", "somewhere", "soon", "sorry",
               "specified", "specify", "specifying", "still", "sub", "such",
               "sup", "sure", "t", "t's", "take", "taken", "tell", "tends",
               "th", "than", "thank", "thanks", "thanx", "that", "that's",
               "thats", "the", "their", "theirs", "them", "themselves",
               "then", "thence", "there", "there's", "thereafter", "thereby",
               "therefore", "therein", "theres", "thereupon", "these", "they",
               "they'd", "they'll", "they're", "they've", "think", "third",
               "this", "thorough", "thoroughly", "those", "though", "three",
               "through", "throughout", "thru", "thus", "to", "together",
               "too", "took", "toward", "towards", "tried", "tries", "truly",
               "try", "trying", "twice", "two", "u", "un", "under",
               "unfortunately", "unless", "unlikely", "until", "unto", "up",
               "upon", "us", "use", "used", "useful", "uses", "using",
               "usually", "uucp", "v", "value", "various", "very", "via",
               "viz", "vs", "w", "want", "wants", "was", "wasn't", "way",
               "we", "we'd", "we'll", "we're", "we've", "welcome", "well",
               "went", "were", "weren't", "what", "what's", "whatever",
               "when", "whence", "whenever", "where", "where's", "whereafter",
               "whereas", "whereby", "wherein", "whereupon", "wherever",
               "whether", "which", "while", "whither", "who", "who's",
               "whoever", "whole", "whom", "whose", "why", "will", "willing",
               "wish", "with", "within", "without", "won't", "wonder",
               "would", "would", "wouldn't", "x", "y", "yes", "yet", "you",
               "you'd", "you'll", "you're", "you've", "your", "yours",
               "yourself", "yourselves", "z", "zero", "quot"]

ps = PorterStemmer()
ENG_STOP_WORDS_SET = set()
for word in EngStopWord:
    word = word.lower().strip()
    word = ps.stem(word)
    ENG_STOP_WORDS_SET.add(word)


def process_xml_data():
    file_columns = ['project', 'bug_id', 'severity', 'priority']
    label_res_data = []

    for doc in doc_name:
        xml_filename = "F:/Exp2/metadata/" + doc + ".xml"
        with open(xml_filename, 'r', errors='ignore') as f:
            content = f.readline()

            while content != '':
                # 首尾完整
                if "<item>" in content:
                    try:
                        bug_id = re.findall('<id>(.*?)</id>', content)[0]
                        bug_priority = re.findall('<priority>(.*?)</priority>', content)[0]
                        bug_severity = re.findall('<bug_severity>(.*?)</bug_severity>', content)[0]
                    except:
                        content = f.readline()
                        continue

                    # low label
                    priority_label = 0
                    # non_severe label
                    severity_label = 0

                    # drop the bug report
                    if bug_severity == "normal":
                        severity_label = -1
                    # severe label
                    elif bug_severity == "blocker" or bug_severity == "critical" or bug_severity == "major":
                        severity_label = 1

                    # high label
                    if bug_priority == 'P1' or bug_priority == 'P2':
                        priority_label = 2
                    # medium label
                    elif bug_priority == 'P3':
                        priority_label = 1

                    label_res_data.append([doc, bug_id, severity_label, priority_label])
                content = f.readline()

    res = pd.DataFrame(label_res_data, columns=file_columns)
    res.to_csv('./label_data/severity_priority_labels.csv', index=False)


# 根据包含的关键词数量筛选关键语句
def select_key_sentences(remove_strategy, context_strategy, context_num, k_value, stack_trace_strategy):
    res_data = []
    file_prefix = "F:/dataset/the_dataset_of_six_open_source_Java_projects/dataset/"

    for doc in doc_name:
        filename = file_prefix + doc + ".xml"
        dom_tree = parse(filename)
        root_node = dom_tree.documentElement

        keyword_filename = "F:/KBL_dataset/KBL/Refinement/" + doc + "_refine.csv"
        keyword_dataset = pd.read_csv(keyword_filename)
        keyword_bug_id = keyword_dataset['BugId'].tolist()
        keywords_list = keyword_dataset['Keyword'].tolist()
        # print(keywords_list)

        bugs = root_node.getElementsByTagName("table")
        for bug in bugs:
            fields = bug.getElementsByTagName("column")

            bug_id = ""
            summary = ""
            description = ""
            for field in fields:
                name = field.getAttribute("name")
                if name == "bug_id":
                    bug_id = str(field.childNodes[0].data)
                elif name == "summary":
                    summary = str(field.childNodes[0].data)
                    summary = summary.split(' ', 2)[2].strip()
                    if not summary.endswith("."):
                        summary += '.'
                elif name == "description":
                    if len(field.childNodes) > 0:
                        description = str(field.childNodes[0].data)

            if bug_id != "":
                print(doc + ": " + bug_id)
                # last_letter = summary[-1:]
                # if "A" <= last_letter <= "Z" or "a" <= last_letter <= "z":
                #     text = summary + ". " + description
                # else:
                #     text = summary + " " + description
                if description != "":
                    text = summary + " " + description
                else:
                    text = summary

                text = text.strip().replace('\n', ' ').replace('\r', ' ')
                # if config.remove_stack_trace:
                if remove_strategy:
                    text, _ = process_stack_trace(text, stack_trace_strategy)
                # 句子切分
                sentences = sent_tokenize(text)
                # 句子内的单词切分
                # 用于输入到语言模型中的句子
                processed_sentences_tmp = [split_camel_case(sent) for sent in sentences]
                processed_sentences = []
                for s in processed_sentences_tmp:
                    if s.strip() != '':
                        processed_sentences.append(s)
                # 用于查找句子中是否包含关键词的句子
                preprocessed_sentences = [preprocess(sent) for sent in processed_sentences]
                # 两个列表的长度相等
                assert len(preprocessed_sentences) == len(processed_sentences)

                # 获取关键词集合的索引
                try:
                    keyword_idx = keyword_bug_id.index(int(bug_id))
                except:
                    continue

                key_sentences = []
                sentence_key = dict()
                for idx in range(len(preprocessed_sentences)):
                    keywords = set(keywords_list[keyword_idx].split(" "))
                    # print(keywords)
                    # 统计句子中包含的句子数量
                    count = is_contain_keywords(keywords, preprocessed_sentences[idx])
                    # 记录
                    sentence_key[idx] = count
                    # res_data.append([bug_id, processed_sentences[idx], count])

                if len(preprocessed_sentences) == 1:
                    key_sentences = processed_sentences
                    # key_sentences = preprocessed_sentences
                else:
                    # 根据关键词数量降序排列
                    lis = sorted(sentence_key.items(), key=lambda kv: kv[1], reverse=True)
                    idx = 0
                    key_ids = []
                    # 取含关键词数量最多的K个关键语句
                    while idx < len(lis):
                        # print(lis[idx])
                        # 判断语句中的关键词数量，若小于等于0，则跳出循环
                        if lis[idx][1] <= 0:
                            break
                        # 控制句子中词语的数量
                        else:
                            sentence = processed_sentences[lis[idx][0]]
                            # 筛去长度不足5个词语的句子
                            if len(sentence.strip().split(' ')) < 5:
                                idx += 1
                                continue
                            if sentence.strip().endswith('.'):
                                processed_sentences[lis[idx][0]] = sentence.strip()[: -1]
                            key_ids.append(lis[idx][0])
                            idx += 1

                        # if len(key_ids) >= config.K:
                        if len(key_ids) >= k_value:
                            break

                    # if config.GET_CONTEXT:
                    if context_strategy:
                        tmp_ids = key_ids
                        key_ids = set()
                        for s_id in tmp_ids:
                            # if config.get_before:
                            for con in range(1, context_num + 1):
                                if s_id - con > 0 and len(sentences[s_id - con].strip().split(' ')) >= 4:
                                    key_ids.add(s_id - con)
                            key_ids.add(s_id)
                            # if config.get_after:
                            for con in range(1, context_num + 1):
                                if s_id + con < len(processed_sentences) and len(sentences[s_id + con].strip().split(' ')) >= 4:
                                    key_ids.add(s_id + con)

                    # 按原始句子的相对顺序排列
                    key_ids = list(key_ids)
                    key_ids.sort()
                    for s_id in key_ids:
                        # 添加未经预处理的原始文本
                        key_sentences.append(processed_sentences[s_id])

                key_text = []
                for sent in key_sentences:
                    # sent = simple_preprocess(sent)
                    # print(sent)
                    key_text.append(sent)
                key_text = '. '.join(key_text).strip()
                res_data.append([bug_id, key_text])

    res = pd.DataFrame(res_data, columns=['bug_id', 'text'])
    # res = pd.DataFrame(res_data, columns=['bug_id', 'sentence', 'keyword_count'])
    res.drop_duplicates(keep='first', inplace=True)
    res.to_csv('./label_data/sentences.csv', index=False)


# 处理词语，将复合词语划分开
def split_camel_case(text):
    # print(text)
    word_buf = []
    content_buf = []

    for character in text:
        if ('a' <= character <= 'z') or ('A' <= character <= 'Z'):
            word_buf.append(character)
            # print(character)
            continue

        length = len(word_buf)
        if length != 0:
            buf_idx = 0
            buf_next_idx = 1
            word_start_idx = 0
            # 复合词拆分
            buf = split(buf_idx, buf_next_idx, word_start_idx, length, word_buf)
            # print(buf)
            content_buf.extend(buf)
        # word_buf.clear()

    # 处理最后一个单词
    length = len(word_buf)
    if length != 0:
        word_start_idx = 0
        buf_idx = 0
        buf_next_idx = 1
        content_buf.extend(split(buf_idx, buf_next_idx, word_start_idx, length, word_buf))

    return ' '.join(content_buf).strip()


# 切分复合词语
def split(buf_idx, buf_next_idx, word_start_idx, length, word_buf):
    # print(word_buf)
    content_buf = []
    while buf_idx < length - 1:
        first = word_buf[buf_idx]
        second = word_buf[buf_next_idx]
        if ('A' <= first <= 'Z') and ('a' <= second <= 'z'):
            content = ''.join(word_buf[word_start_idx: buf_idx]).strip()
            if content.strip() != "" and len(content) >= 2:
                content_buf.append(content)

            word_start_idx = buf_idx
            buf_idx += 1
            buf_next_idx += 1
            continue

        if ('a' <= first <= 'z') and ('A' <= second <= 'Z'):
            content = ''.join(word_buf[word_start_idx: buf_next_idx]).strip()
            if content != "" and len(content) >= 2:
                content_buf.append(content)

            word_start_idx = buf_next_idx
        buf_idx += 1
        buf_next_idx += 1

    if word_start_idx < length:
        content = ''.join(word_buf[word_start_idx:]).strip()
        if content != "" and len(content) >= 2:
            content_buf.append(content)
    word_buf.clear()

    return content_buf


def preprocess(text):
    res = []
    token_list = text.split(' ')
    for token in token_list:
        # token = token.lower()
        token = ps.stem(token.lower())
        if token not in ENG_STOP_WORDS_SET:
            res.append(token)

    return ' '.join(res).strip()


def simple_preprocess(text):
    # print(text)
    tokens = str(text).strip().split()
    res = []
    for token in tokens:
        token = token.strip()
        if token == '' or len(token) < 2:
            continue
        buf = []
        for letter in token:
            # print(letter)
            if letter.isalpha():
                buf.append(letter)

        # print(buf)
        if len(buf) >= 2:
            res.append(''.join(buf))

    # print(res)
    return ' '.join(res).strip()


def is_contain_keywords(keyword_set, sentence):
    contained_keyword = set()
    token_list = sentence.split(' ')
    for token in token_list:
        if token in keyword_set:
            contained_keyword.add(token)

    return len(contained_keyword)


# 划分微调数据集的训练集和验证集
def split_train_valid_data(data_file):
    data_set = pd.read_csv(data_file)
    data_set = shuffle(data_set, random_state=config.init_seed)
    train_size = int(len(data_set) * 0.9)
    train_data = data_set[: train_size]
    valid_data = data_set[train_size:]

    train_data.to_csv('./input_data/train_sentences.csv', index=False)
    valid_data.to_csv('./input_data/valid_sentences.csv', index=False)


# def split_train_test_data(bug_list, category_num, task):
#     # 80%数据集进行训练，10%数据集进行验证，10%数据集进行测试
#     train_set = []
#     valid_set = []
#     test_set = []
#
#     for idx in range(category_num):
#         # 挑选特定标签的数据
#         corr_bug_list = bug_list[bug_list[task] == idx]['bug_id'].tolist()
#         # 打乱顺序
#         random.shuffle(corr_bug_list)
#         # 大小划分
#         train_size = int(len(corr_bug_list) * 0.8)
#         valid_size = int((len(corr_bug_list) - train_size) / 2)
#
#         train_set += corr_bug_list[: train_size]
#         valid_set += corr_bug_list[train_size: train_size + valid_size]
#         test_set += corr_bug_list[train_size + valid_size:]
#
#     return train_set, valid_set, test_set
#
#
# def split_label_data(filepath, category_num, task):
#     complete_dataset = pd.read_csv(filepath)
#     train_id_list, valid_id_list, test_id_list = split_train_test_data(complete_dataset, category_num, task)
#
#     # 筛选出bug id对应的数据行
#     train_dataset = complete_dataset[complete_dataset['bug_id'].isin(train_id_list)]
#     valid_dataset = complete_dataset[complete_dataset['bug_id'].isin(valid_id_list)]
#     test_dataset = complete_dataset[complete_dataset['bug_id'].isin(test_id_list)]
#
#     res_file_prefix = "./split_data/" + task
#     train_dataset.to_csv(res_file_prefix + "_train.csv", index=False)
#     valid_dataset.to_csv(res_file_prefix + "_valid.csv", index=False)
#     test_dataset.to_csv(res_file_prefix + "_test.csv", index=False)
#
#
# def process_severity_data():
#     # 过滤掉标签为-1的数据
#     filepath = "./label_data/severity_priority_labels.csv"
#     dataset = pd.read_csv(filepath, usecols=['project', 'bug_id', 'severity'])
#     dataset = dataset[dataset['severity'] != -1]
#     dataset.to_csv("./label_data/severity_labels.csv", index=False)
#
#
# def process_priority_data():
#     # 将priority数据单独拎出来
#     filepath = "./label_data/severity_priority_labels.csv"
#     dataset = pd.read_csv(filepath, usecols=['project', 'bug_id', 'priority'])
#     dataset.to_csv("./label_data/priority_labels.csv", index=False)
#


# 将打标签后的句子与任务标签做匹配
def combine_label_data_and_sentences(task):
    label_file = config.label_data_dir + task + "_labels.csv"
    label_dataset = pd.read_csv(label_file)

    doc = ""
    doc_dataset = None
    data_file_prefix = 'F:/dataset/the_dataset_of_six_open_source_Java_projects/dataset/'

    columns = ['project', 'bug_id', 'full_text', task]
    data = []

    for idx, row in label_dataset.iterrows():
        project = row['project']
        # 读取项目的数据信息
        if project != doc:
            doc_dataset = pd.read_csv(data_file_prefix + project + '.csv')
            doc = project

        bug_id = row['bug_id']
        doc_line = doc_dataset.loc[doc_dataset['bug_id'] == int(bug_id)]
        # print(doc_line)
        # 获取summary 并去除前缀
        summary = str(doc_line['summary'].values[0]).strip()[4 + len(str(bug_id)):].strip()
        desc = str(doc_line['description'].values[0]).strip()
        label = row[task]
        # 文本拼接
        if summary.endswith('.'):
            summary = summary + ' '
        else:
            summary = summary + '. '
        if desc != 'nan':
            text = summary + desc
        else:
            text = summary.strip()
        text = text.strip().replace('\n', ' ').replace('\r', ' ')
        # 句子切分
        sentences = sent_tokenize(text)
        # 句子中的词语处理
        processed_sentences_tmp = [split_camel_case(sent) for sent in sentences if sent.strip() != ""]
        processed_sentences = []
        for sentence in processed_sentences_tmp:
            if sentence.strip() != '':
                processed_sentences.append(sentence)
        processed_text = '. '.join(processed_sentences)

        data.append([project, bug_id, processed_text, label])

    res = pd.DataFrame(data, columns=columns)
    res.to_csv(config.task_specific_data_dir + task + '.csv', index=False)
#
#
# def combine_data(task):
#     data_types = ['train', 'valid', 'test']
#     for data_type in data_types:
#         combine_label_data_and_sentences(task, data_type)


def combine_label_data_with_key_sentences(task):
    label_data = pd.read_csv(config.label_data_dir + task + "_labels.csv")
    key_sentences_data = pd.read_csv(config.label_data_dir + 'sentences.csv')

    columns = ['project', 'bug_id', 'full_text', task]
    data = []

    for idx,row in label_data.iterrows():
        bug_id = row['bug_id']
        key_sentence = key_sentences_data.loc[key_sentences_data['bug_id'] == bug_id]
        if len(key_sentence) == 0:
            continue

        data.append([row['project'], bug_id, key_sentence['text'].values[0], row[task]])

    res = pd.DataFrame(data, columns=columns)
    res.to_csv(config.task_specific_data_dir + task + '_key_sentences.csv', index=False)


# 下游任务数据集划分训练集和测试集
def split_task_train_test(task, category_num, use_key=False):
    if use_key:
        task_file = config.task_specific_data_dir + task + '_key_sentences.csv'
    else:
        task_file = config.task_specific_data_dir + task + ".csv"
    task_data = pd.read_csv(task_file)
    cate_size = []
    train_set = []
    test_set = []

    for idx in range(category_num):
        corr_bug_list = task_data[task_data[task] == idx].values.tolist()
        # 随机打乱
        random.seed(config.init_seed)
        random.shuffle(corr_bug_list)
        # 80%为训练集，20%为测试集
        train_size = int(0.8 * len(corr_bug_list))
        train_set += corr_bug_list[: train_size]
        test_set += corr_bug_list[train_size:]
        # 记录各个类别对应的训练集大小
        cate_size.append(train_size)

    random.shuffle(train_set)
    random.shuffle(test_set)

    train_data = pd.DataFrame(train_set, columns=['project', 'bug_id', 'full_text', task])
    test_data = pd.DataFrame(test_set, columns=['project', 'bug_id', 'full_text', task])
    if use_key:
        train_data.to_csv(config.task_specific_data_dir + task + '_key_sentences_train.csv', index=False)
        test_data.to_csv(config.task_specific_data_dir + task + '_key_sentences_test.csv', index=False)
    else:
        train_data.to_csv(config.task_specific_data_dir + task + '_train.csv', index=False)
        test_data.to_csv(config.task_specific_data_dir + task + '_test.csv', index=False)


def get_full_fine_tune_data():
    BR_data = pd.read_csv(config.task_specific_data_dir + 'priority.csv', usecols=['bug_id', 'full_text'])
    all_ids = BR_data['bug_id'].tolist()
    all_text = BR_data['full_text'].tolist()
    train_ids = pd.read_csv(config.fine_tune_train_data_dir, usecols=['bug_id'])['bug_id'].tolist()

    train_text = []
    valid_text = []
    for train_id in train_ids:
        idx = all_ids.index(train_id)
        train_text.append([train_id, all_text[idx]])

    valid_ids = list(set(all_ids).difference(set(train_ids)))
    for valid_id in valid_ids:
        idx = all_ids.index(valid_id)
        valid_text.append([valid_id, all_text[idx]])

    res = pd.DataFrame(train_text, columns=['bug_id', 'text'])
    res.to_csv(config.full_fine_tune_train_data_dir, index=False)
    res = pd.DataFrame(valid_text, columns=['bug_id', 'text'])
    res.to_csv(config.full_fine_tune_test_data_dir, index=False)


# 划分句子，打标签（是否包含关键词）
def label_sentences(remove_strategy, stack_trace_strategy):
    # 只用于获取缺陷报告的项目名与id
    label_file = config.label_data_dir + "reopen_labels.csv"
    label_dataset = pd.read_csv(label_file)

    # 获取原始的缺陷报告数据
    doc = ""
    doc_dataset = None
    data_file_prefix = 'F:/dataset/the_dataset_of_six_open_source_Java_projects/dataset/'

    # 获取缺陷报告对应的关键词
    keyword_filename = "F:/KBL_dataset/KBL/Refinement/"
    keyword_bug_id = []
    keywords_list = []

    columns = ['project', 'bug_id', 'full_text', 'label']
    data = []
    text_columns = ['project', 'bug_id', 'pure_text', 'stack_trace']
    text_data = []

    for idx, row in label_dataset.iterrows():
        project = row['project']
        # 读取项目的数据信息（原始数据&关键词）
        if project != doc:
            doc_dataset = pd.read_csv(data_file_prefix + project + '.csv')
            doc = project
            keyword_dataset = pd.read_csv(keyword_filename + project + "_refine.csv")
            keyword_bug_id = keyword_dataset['BugId'].tolist()
            keywords_list = keyword_dataset['Keyword'].tolist()

        bug_id = row['bug_id']
        doc_line = doc_dataset.loc[doc_dataset['bug_id'] == int(bug_id)]

        # 查找是否存在相应的关键词数据，不存在则处理下一条缺陷报告
        try:
            keyword_idx = keyword_bug_id.index(int(bug_id))
        except:
            continue
        # 关键词集合
        keywords = set(str(keywords_list[keyword_idx]).strip().split(' '))

        # 获取summary 并去除前缀
        summary = str(doc_line['summary'].values[0]).strip()[4 + len(str(bug_id)):].strip()
        desc = str(doc_line['description'].values[0]).strip()
        # 文本拼接
        # 将摘要与详情用句号分隔开
        if summary.endswith('.'):
            summary = summary.strip() + ' '
        else:
            summary = summary + '. '
        # 判断详情是否为空
        if desc != 'nan':
            text = summary + desc
        else:
            text = summary.strip()

        print(bug_id)

        # if config.remove_stack_trace:
        if remove_strategy:
            text, stack_traces = process_stack_trace(text, stack_trace_strategy)
            text_data.append([project, bug_id, text, '\n'.join(stack_traces)])
        # 去除换行和缩进符
        text = text.strip().replace('\n', ' ').replace('\r', ' ')
        # text = re.sub(r'\s+', ' ', text)

        # 句子切分
        sentences = sent_tokenize(text)
        # print(sentences)

        # 句子中的词语处理
        processed_sentences = [split_camel_case(sent) for sent in sentences if sent != ""]
        preprocessed_sentences = [preprocess(sent) for sent in processed_sentences]
        label = []

        assert len(preprocessed_sentences) == len(processed_sentences) == len(sentences)
        used_sents = []
        for idx in range(len(preprocessed_sentences)):
            single_sent = processed_sentences[idx]
            # single_sent = single_sent.replace('. ', ', ')
            # if single_sent.strip() == '':
            #     continue
            if len(single_sent.strip().split(' ')) < 5:
                continue
            key_count = is_contain_keywords(keywords, single_sent)
            if key_count <= 0:
                label.append("0")
            else:
                label.append(str(key_count))
                # label.append("1")
            used_sents.append(single_sent)

        assert len(used_sents) == len(label)
        # used_sents = [simple_preprocess(sent) for sent in used_sents]
        full_text = '. '.join(used_sents).strip()
        label = ' '.join(label).strip()

        data.append([project, bug_id, full_text, label])

    res = pd.DataFrame(data, columns=columns)
    res.to_csv(config.label_data_dir + 'sent_classify.csv', index=False)

    # res = pd.DataFrame(text_data, columns=text_columns)
    # res.to_csv(config.label_data_dir + 'text_stack_trace.csv', index=False)


# 抽取缺陷报告文本中的堆栈语句
def process_stack_trace(text, stack_trace_strategy):
    stack_trace_lines = []
    # 正则表达式匹配堆栈信息
    pattern = re.compile(r'(((?P<exception_name>(\b[A-Za-z]+)?Exception)\sin\sthread)|(org\.)|(sun\.)|(java\.)|(at\s(\b[A-Za-z]+)\.(\b[A-Za-z]+)))([^\s]+)\.([^\s]+)((\((?P<file_name>(.+)\.(java|aj)(:\d+)*)\))|(\(Unknown Source\))|(\(Native Method\)))')
    matches = pattern.search(text)

    exception_pattern = re.compile(r'(\b[A-Za-z]+)?(Exception|Error)\s')
    file_name_pattern = re.compile(r'\bat\s+(?P<class_name>[\w.$]+)\.(?P<method_name>\w+)\((?P<file_name>[^:(]+)(:(?P<line_number>\d+))?\)')

    template = "{} method of {} class encountered an {} in {} file {}."

    while matches is not None:
        print(matches.group())
        without_st_text = text.replace(matches.group(), '')
        # 正则表达式匹配抛出了什么Exception
        exception_match = exception_pattern.search(matches.group())
        exception_name = 'Exception'

        # if config.replace_strategy == 'exception':
        if stack_trace_strategy == 'exception':
            if exception_match is not None:
                # exception_name = exception_match.group()
                # print(exception_match.group())
                stack_trace_lines.append(exception_match.group())
                # 判断Exception是否在剔除后的文本中出现过，若出现过，则不加入，否则将Exception的名字添加到stack trace原有的位置
                # text = text.replace(matches.group(), ".")
                text = text.replace(matches.group(), exception_match.group() + ". ")
            else:
                text = text.replace(matches.group(), '.')
            stack_trace_lines.append(matches.group())
        elif stack_trace_strategy == 'template':
            # 获取exception名和异常位置信息
            if exception_match is not None:
                exception_name = exception_match.group().strip()
            file_data = file_name_pattern.findall(matches.group())
            if len(file_data) > 0:
                file_data = file_data[0]
                print(file_data)
                method_name = file_data[1].strip()

                file_name_data = file_data[2].split('.')
                file_name = file_name_data[0].strip()
                if len(file_name_data) > 2:
                    file_type = file_data[2].split('.')[-1].strip()
                elif len(file_name_data) == 1:
                    file_type = ''
                else:
                    file_type = file_data[1].strip()

                class_name = file_data[0].split('.')[-1].strip()
                template = template.format(method_name, class_name, exception_name, file_type, file_name).replace('  ', ' ')
            else:
                template = "The {} is encountered. ".format(exception_name)
            text = text.replace(matches.group(), template)
            print(template)
            stack_trace_lines.append(template)
        elif stack_trace_strategy == 'remove':
            text = text.replace(matches.group(), '')

        # 从当前搜索到的位置继续往后找，看是否还有堆栈信息内容
        matches = pattern.search(text, matches.end())

    return text, stack_trace_lines


def get_classify_train_test():
    origin_data = pd.read_csv(config.label_data_dir + 'sent_classify.csv', usecols=['bug_id', 'full_text', 'label'])
    train_ids = pd.read_csv(config.fine_tune_train_data_dir, usecols=['bug_id'])['bug_id'].tolist()
    all_ids = origin_data['bug_id'].tolist()
    origin_sents = origin_data['full_text'].tolist()
    origin_labels = origin_data['label'].tolist()

    train_data = []
    for train_id in train_ids:
        idx = all_ids.index(train_id)
        if type(origin_sents[idx]) is float and math.isnan(origin_sents[idx]):
            continue
        train_data.extend(split_sentences_labels(origin_sents[idx], origin_labels[idx]))

    valid_ids = list(set(all_ids).difference(set(train_ids)))
    valid_data = []
    for valid_id in valid_ids:
        idx = all_ids.index(valid_id)
        if type(origin_sents[idx]) is float and math.isnan(origin_sents[idx]):
            continue
        valid_data.extend(split_sentences_labels(origin_sents[idx], origin_labels[idx]))

    res = pd.DataFrame(train_data, columns=['text', 'label'])
    res.to_csv(config.classify_train_data_dir, index=False)
    res = pd.DataFrame(valid_data, columns=['text', 'label'])
    res.to_csv(config.classify_test_data_dir, index=False)


def split_sentences_labels(sent_list, label_list):
    data = []
    sent_list = sent_list.split('. ')
    label_list = label_list.split(' ')

    for idx in range(len(sent_list)):
        if len(sent_list[idx].split(' ')) < 4:
            continue
        data.append([sent_list[idx], label_list[idx]])

    return data


def get_key_sentences(context_strategy, context_num, k_value):
    sent_file = config.label_data_dir + "sent_classify.csv"
    sent_dataset = pd.read_csv(sent_file)

    data = []
    columns = ['bug_id', 'text']
    for idx, row in sent_dataset.iterrows():
        bug_id = row['bug_id']
        try:
            sent_list = row['full_text'].split('. ')
        except AttributeError:
            continue
        try:
            sent_label = row['label'].split(' ')
        except AttributeError:
            # label为空的情况
            continue
        # 所有句子都不包含关键词的情况
        contains_key_sentence = False
        for la in sent_label:
            if la != '0':
                contains_key_sentence = True
                break
        if not contains_key_sentence:
            continue

        print("{} --- {}".format(len(sent_label), len(sent_list)))
        assert len(sent_label) == len(sent_list)
        key_sentences = get_key_sentences_with_k(sent_list, sent_label, context_strategy, context_num, k_value)
        # key_sentences = get_key_sentences_with_k(sent_list, sent_label, config.K)

        data.append([bug_id, ' '.join(key_sentences).strip()])

    res = pd.DataFrame(data, columns=columns)
    res.to_csv(config.label_data_dir + "sentences.csv", index=False)


def get_key_sentences_with_k(sent_list, sent_labels, context_strategy, context_num, k):
    find_k = False
    max_key_count = 0
    # 先遍历一遍，看是否有符合条件的句子，若没有，则k退化为句子中包含的最多的关键词数量
    for key_count in sent_labels:
        if int(key_count) >= k:
            find_k = True
            break
        else:
            # 记录最大值
            if int(key_count) > max_key_count:
                max_key_count = int(key_count)

    # k值退化
    if not find_k:
        k = max_key_count

    key_sentences = []
    # 用于记录句子是否被添加到key_sentences中
    added_sent = [0] * len(sent_list)
    assert len(sent_labels) == len(sent_list)
    for index in range(len(sent_list)):
        sent = sent_list[index]
        if int(sent_labels[index]) >= k:
            # 判断是否需要添加上下文
            if context_strategy:
                # if config.get_before:
                for con in range(1, context_num + 1):
                    if index - con > 0 and added_sent[index - con] != 1 and len(sent_list[index - con].strip().split(' ')) >= 4:
                        key_sentences.append(sent_list[index - con])
                        added_sent[index - con] = 1
            if added_sent[index] == 0:
                key_sentences.append(sent_list[index])
                added_sent[index] = 1
            if context_strategy:
                # if config.get_after:
                for con in range(1, context_num):
                    if index + con < len(sent_list) and added_sent[index + con] == 0 and len(sent_list[index + con].strip().split(' ')) >= 4:
                        key_sentences.append(sent_list[index + con])
                        added_sent[index + con] = 1
                # if index > 0 and added_sent[index - 1] == 0:
                #     key_sentences.append(sent_list[index - 1])
                #     added_sent[index - 1] = 1
                # # 判断是否为最后一个句子
                # if index < len(sent_list) - 1:
                #     key_sentences.append(sent_list[index + 1])
                #     added_sent[index + 1] = 1
            if added_sent[index] == 0:
                key_sentences.append(sent)

    return key_sentences


# file = "./label_data/severity_labels.csv"
# task_type = "severity"
# cate_num = 2
# use_key_sentences = True
# split_label_data(file, cate_num, task_type)

# combine_data(task_type)

# if config.sentence_strategy == 'at least k':
#     label_sentences(config.remove_stack_trace, config.replace_strategy)
#     get_key_sentences(context_strategy=config.GET_CONTEXT, context_num=config.context_n, k_value=config.K)
# elif config.sentence_strategy == 'top-k':
#     select_key_sentences(context_strategy=config.GET_CONTEXT, context_num=config.context_n, k_value=config.K, remove_strategy=config.remove_stack_trace, stack_trace_strategy=config.replace_strategy)
# file = './label_data/sentences.csv'
# split_train_valid_data(file)
# combine_label_data_with_key_sentences(task_type)
# combine_label_data_and_sentences(task_type)
# split_task_train_test(task_type, cate_num, use_key_sentences)
# split_task_train_test(task_type, cate_num, False)
# get_full_fine_tune_data()
# label_sentences()
# get_classify_train_test()
