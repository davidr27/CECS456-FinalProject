# CECS456-FinalProject
Original Model - PyCharm
To train the model, please python main.py first. Then this will create model_structure.json and model_weights.h5 files
To make the predictions, run the predictions.py file. This python file will use the model_structure.json and model_weights.h5 files
On line #20 change the name of the ".jpg" to one of the images provided, Ex: img = image.load_img("Buddy.jpg", target_size=(128, 128))
The program will give you the likelihood of what kind of animal it is.
