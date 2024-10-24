# TODO: Get current test task from base.yaml
# TODO: Read task hyperparams from config/test/TEST_NAME/test_info.yaml into a dictionary
# TODO: Read model hyperparams (for specific task) from config/test/TEST_NAME/MODEL_NAME.yaml into a dictionary
# TODO: Read model weights from models/MODEL_NAME/weights/TRAIN_NAME (or models/MODEL_NAME/weights/initial)
# TODO: Apply model to test set
# TODO: Save task hyperparams from config/test/TEST_NAME/test_info.yaml and current time as test_info.json in both results/test/TEST_NAME and models/MODEL_NAME/weights/TEST_NAME
# TODO: Save model hyperparams from config/test/TEST_NAME/MODEL_NAME.yaml and current time as MODEL_NAME.json in both results/test/TEST_NAME/MODEL_NAME and models/MODEL_NAME/weights/TEST_NAME
# TODO: Dump results in the results/test/TEST_NAME/MODEL_NAME (separate folders for outputs, plots, and metrics)