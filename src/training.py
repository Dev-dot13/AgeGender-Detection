from data_preprocessing import prepare_datasets
from model_training import create_multi_task_model, train_model

# Prepare data
(X_train, y_age_train, y_gender_train), (X_val, y_age_val, y_gender_val), _ = prepare_datasets('data/train/crop_part1')

# Create and train model
model = create_multi_task_model()
model, history = train_model(model, X_train, y_age_train, y_gender_train, X_val, y_age_val, y_gender_val, epochs=10)

# Save model
model.save('models/vgg16_age_gender.h5')