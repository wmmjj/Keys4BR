import json
import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from peft import LoraConfig, AdaLoraConfig, PromptTuningConfig, get_peft_model, PeftType, TaskType
# from torch.utils.tensorboard import SummaryWriter
from transformers import RobertaTokenizer, RobertaForMaskedLM, AutoTokenizer, DebertaForMaskedLM, DebertaTokenizer
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling,TrainerCallback
from adapters import AutoAdapterModel, AdapterTrainer, AdapterType

import config


class BugReportMLMDataset(Dataset):
    def __init__(self, data_dir, tokenizer, fine_tune_strategy):
        self.data_dir = data_dir
        self.tokenizer = tokenizer

        self.data = pd.read_csv(data_dir)
        self.data.dropna(inplace=True)
        self.fine_tune_strategy = fine_tune_strategy

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_row = self.data.iloc[idx]
        text = data_row['text']
        max_len = 512
        if self.fine_tune_strategy == 'prompt tuning':
            max_len = 502
        inputs = self.tokenizer.encode_plus(text, add_special_tokens=True, pad_to_max_length='max_length', padding=True, truncation=True, max_length=512)
        input_ids = torch.tensor(inputs['input_ids'])
        attention_mask = torch.tensor(inputs['attention_mask'])
        labels = input_ids.clone()
        # 避免对损失函数的计算产生贡献
        labels[labels == 0] = -100
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


class CustomCallback(TrainerCallback):
    def __init__(self, trainer):
        super().__init__()
        self._trainer = trainer

    # def on_step_end(self, args, state, control, **kwargs):
    #     step = self._trainer.state.global_step
    #     epoch = self._trainer.state.epoch
    #     if step < 5000:
    #         if step % 100 == 0:
    #             save_checkpoint(args, self._trainer, epoch, step)

    # def on_epoch_end(self, args, state, control, **kwargs):
    #     epoch = self._trainer.state.epoch
    #     step = self._trainer.state.global_step
    #     save_checkpoint(args, self._trainer, epoch, step)


# 保存checkpoint
def save_checkpoint(args, trainer, epoch, step):
    checkpoint_folder = os.path.join(args.output_dir, f'./test-{step}')
    os.makedirs(checkpoint_folder, exist_ok=True)
    trainer.save_model(checkpoint_folder)
    trainer_state = trainer.state.__dict__
    with open(os.path.join(checkpoint_folder, "trainer_state.json"), 'w') as fp:
        json.dump(trainer_state, fp, indent=2)


# 设置随机种子
def init_seed(seed, reproducibility):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def train(fine_tune_strategy=None, save_file_name=None):
    device = config.device
    # 预训练模型
    # PRETRAINED_MODEL = './deberta-base'
    PRETRAINED_MODEL = './roberta-base'
    # 加载tokenizer
    # TOKENIZER = DebertaTokenizer.from_pretrained(PRETRAINED_MODEL)
    tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED_MODEL)

    # model = DebertaForMaskedLM.from_pretrained(PRETRAINED_MODEL)
    model = RobertaForMaskedLM.from_pretrained(PRETRAINED_MODEL)
    model.to(device)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.size())
    # return

    # 完整文本数据集
    # train_dataset = BugReportMLMDataset(config.full_fine_tune_train_data_dir, tokenizer)
    # test_dataset = BugReportMLMDataset(config.full_fine_tune_test_data_dir, tokenizer)
    # res_path = config.fine_tune_model + 'full_text_fine_tune'

    # 关键语句数据集
    train_dataset = BugReportMLMDataset(config.fine_tune_train_data_dir, tokenizer, fine_tune_strategy)
    test_dataset = BugReportMLMDataset(config.fine_tune_test_data_dir, tokenizer, fine_tune_strategy)
    if save_file_name is not None:
        res_path = config.fine_tune_model + 'key_sentences_fine_tune_' + save_file_name
    else:
        res_path = config.fine_tune_model + 'key_sentences_fine_tune_top_1_context_1_exception'
    # res_path = config.fine_tune_model + 'key_sentences_fine_tune_at_least_2'

    # if os.path.exists(res_path):
    #     print('the model is already existed')
    #     return

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=config.fine_tune_model,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=8,
        # lr_scheduler_type="linear",
        # warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=5,  # log every n steps
        load_best_model_at_end=True,  # save/checkpoint the best model_mask at the end of training
        prediction_loss_only=True,
        # metric_for_best_model='eval_loss',  # use loss to evaluate the best model_mask
        save_total_limit=5,  # limit the total amount of checkpoints
        gradient_accumulation_steps=2,
        # number of gradient accumulation steps before performing a backward/update pass.
        per_device_eval_batch_size=8,  # batch size for evaluation
        # evaluation_strategy="epoch",  # evaluation strategy to adopt during training
        save_strategy='no',  # checkpoint save strategy
        # eval_steps=1000,  # evaluation step.
        # save_steps=1000,  # save checkpoint every n steps.
        learning_rate=config.lr,  # learning rate
    )

    if fine_tune_strategy is None or fine_tune_strategy == 'freeze':
        # 不冻结的模型层名
        unfreeze_layers = ['layer.10', 'layer.11', 'cls.predictions']
        # unfreeze_layers = ['layer.10', 'layer.11', 'roberta.pooler', 'classifier.']

        # 冻结其他层参数
        for name, param in model.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
    elif fine_tune_strategy == 'prompt tuning':
        prompt_tuning_config = PromptTuningConfig(
            num_virtual_tokens=10,
            peft_type=PeftType.PROMPT_TUNING,
            task_type=TaskType.CAUSAL_LM,
            tokenizer_name_or_path='./roberta-base',
            # base_model_name_or_path='./roberta-base',
            # token_dim=768
        )
        model = get_peft_model(model, prompt_tuning_config)
    elif fine_tune_strategy == 'lora':
        lora_config = LoraConfig(
            peft_type=PeftType.LORA,
            task_type=TaskType.FEATURE_EXTRACTION,
            # target_modules=['q', 'v'],
            r=1,
            lora_alpha=32,
            lora_dropout=0.01
        )
        model = get_peft_model(model, lora_config)
    elif fine_tune_strategy == 'adalora':
        adalora_config = AdaLoraConfig(
            peft_type=PeftType.ADALORA,
            task_type=TaskType.FEATURE_EXTRACTION,
            r=16,
            lora_alpha=32,
            # target_modules=['q', 'v'],
            lora_dropout=0.01
        )
        model = get_peft_model(model, adalora_config)

    # 训练模型
    model.train().to(device)
    # 使用优化器更新参数
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
    linear_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(len(train_dataset) * 0.1), gamma=0.1)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, linear_scheduler)
    )

    trainer.add_callback(CustomCallback(trainer))

    trainer.train()
    trainer.save_model(res_path)
    # trainer.save_model(config.fine_tune_model + 'model_mask/')
    # trainer.save_model(config.fine_tune_model + 'model_compare')


def adapter_train(writer):

    model = AutoAdapterModel.from_pretrained('roberta-base', local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained('./roberta-base', local_files_only=True)
    adapter = model.load_adapter('./roberta-base-pf-imdb')
    model.active_adapters = adapter
    device = config.device
    model.to(device)

    # for name, param in model.named_parameters():
    #     print(name, param.size())

    # 关键语句数据集
    train_dataset = BugReportMLMDataset(config.fine_tune_train_data_dir, tokenizer)
    test_dataset = BugReportMLMDataset(config.fine_tune_test_data_dir, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=config.fine_tune_model,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=8,
        # warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=5,  # log every n steps
        load_best_model_at_end=True,  # save/checkpoint the best model_mask at the end of training
        prediction_loss_only=True,
        metric_for_best_model='eval_loss',  # use loss to evaluate the best model_mask
        save_total_limit=5,  # limit the total amount of checkpoints
        gradient_accumulation_steps=2,
        # number of gradient accumulation steps before performing a backward/update pass.
        per_device_eval_batch_size=8,  # batch size for evaluation
        evaluation_strategy="epoch",  # evaluation strategy to adopt during training
        save_strategy='epoch',  # checkpoint save strategy
        # eval_steps=1000,  # evaluation step.
        # save_steps=1000,  # save checkpoint every n steps.
        learning_rate=config.lr,  # learning rate
    )

    # model.add_adapter(adapter, AdapterType.text_task)
    model.train_adapter(adapter)
    model.to(device)

    # 冻结模型参数
    for name, param in model.named_parameters():
        param.requires_grad = False

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator
    )

    trainer.add_callback(CustomCallback(trainer))

    trainer.train()
    trainer.save_model(config.fine_tune_model + 'model_adapter')


def main():
    # writer = SummaryWriter('./logs')
    init_seed(config.init_seed, config.reproducibility)
    # adapter_train(writer)
    train()
    # writer.close()


if __name__ == '__main__':
    main()
