# TODO: Get current training task from base.yaml
# TODO: Read task hyperparams from config/train/TRAIN_NAME/train_info.yaml into a dictionary
# TODO: Read model hyperparams (for specific task) from config/train/TRAIN_NAME/MODEL_NAME.yaml into a dictionary
# TODO: Train the model
#  Dump training weights in models/weights/TRAIN_NAME, unless we disable weights saving (in either base config, task info or model config for task)
#  Dump training logs in models/weights/TRAIN_NAME/logs
# TODO: Save task hyperparams from config/train/TRAIN_NAME/train_info.yaml and current time as train_info.json in both results/train/TRAIN_NAME and models/MODEL_NAME/weights/TRAIN_NAME
# TODO: Save model hyperparams from config/train/TRAIN_NAME/MODEL_NAME.yaml and current time as MODEL_NAME.json in both results/train/TRAIN_NAME/MODEL_NAME and models/MODEL_NAME/weights/TRAIN_NAME
# TODO: Dump results in the results/train/TRAIN_NAME/MODEL_NAME (separate folders for outputs, plots, and metrics)