## Evaluate Yolo models
# Generate results from the trained model

Once a yolo model has been trained the results must be generated to evaluate the model.
Set paths to validation data, deploy prototxt and trained caffemodel in `yolo_validate_config.xml`. 
If necessary make changes to other fields. Execute:

`./build/examples/yolo/evaluation/yolo_validate.bin examples/yolo/evaluation/yolo_validate_config.xml`  

# Evaluate trained model
Average Precision is the metric used to evaluate the models. Make necessary changes to `eval_config.xml`.
Execute:
`python evaldet.py eval_config.xml`

