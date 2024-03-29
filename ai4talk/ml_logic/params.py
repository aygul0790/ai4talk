
class Config:
    learning_rate = 0.0005 # 0.0005 previously
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_steps = 2
    batch_size = 2
    num_worker = 12
    num_train_epochs = 20
    gradient_accumulation_steps = 1
    sample_rate = 16000
