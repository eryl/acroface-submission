from torch.optim import AdamW

from acroface.evaluation.performance import HigherIsBetterMetric, LowerIsBetterMetric
from acroface.evaluation.constants import ROC_AUC, BCE
from acroface.models.wrapper import ModelWrapperConfig
from acroface.models.farl import FarlModel, FarlModelConfig
from acroface.models.batch_trained import MiniBatchTrainedModelConfig
from acroface.models.torch_model import DataloaderConfig
from acroface.experiment.hyperparameters import HyperParameterLogUniform, HyperParameterInteger


pretraining_learning_rate = HyperParameterLogUniform(name='pretraining_learning_rate', low=1e-6, high=1e-4)
pretraining_weight_decay = HyperParameterLogUniform(name='pretraining_weight_decay', low=1e-7, high=1e-5)
pretraining_epochs = HyperParameterInteger(name="pretraining_epochs", low=1, high=10)

#learning_rate = 1e-5
learning_rate = HyperParameterLogUniform(name='learning_rate', low=1e-6, high=1e-4)
weight_decay = HyperParameterLogUniform(name='weight_decay', low=1e-7, high=1e-5)
max_epochs = HyperParameterInteger(name="max_epochs", low=1, high=20)

trainer_config = MiniBatchTrainedModelConfig(evaluation_metrics=[HigherIsBetterMetric(ROC_AUC),
                                                                 LowerIsBetterMetric(BCE)], 
                                             max_epochs=max_epochs,
                                             early_stopping_patience=5,
                                             #do_pre_eval=True,
                                             #eval_iterations=10,
                                             )

training_dataloader_config = DataloaderConfig(shuffle=True, num_workers=10, persistent_workers=False)
dev_dataloader_config = DataloaderConfig(shuffle=False, num_workers=10, persistent_workers=False)

model_config = FarlModelConfig(trainer_config=trainer_config, 
                                 batch_size=16,
                                 update_iterations=32,
                                 optim_class=AdamW, 
                                 optim_kwargs={'lr': learning_rate, 'weight_decay': weight_decay},
                                 device='cuda',
                                 train_dataloader_config=training_dataloader_config,
                                 dev_dataloader_config=dev_dataloader_config,
                                 do_frozen_pretraining = True,
                                 frozen_pretraining_optim_class=AdamW,
                                 frozen_pretraining_optim_kwargs = {'lr': pretraining_learning_rate, 'weight_decay': pretraining_weight_decay},
                                 frozen_pretraining_epochs = pretraining_epochs,
                                 
                                   )
                 
wrapper_config = ModelWrapperConfig(model_class=FarlModel, 
                                    model_kwargs={'config': model_config})
