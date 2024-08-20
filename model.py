from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(data="data.yaml", epochs=1)  # train the model

# increase the epoch to make a much better and good model
# the results given the plots as well , see the plot and the loss should decrease wit epochs