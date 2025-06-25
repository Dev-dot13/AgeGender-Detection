from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

def create_multi_task_model(input_shape=(224, 224, 3), age_classes=9, gender_classes=2):
    # Load pre-trained VGG16 without top layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze initial layers
    for layer in base_model.layers[:15]:
        layer.trainable = False
    
    # Add custom layers
    x = Flatten()(base_model.output)
    
    # Age prediction branch
    age_branch = Dense(512, activation='relu')(x)
    age_branch = Dropout(0.5)(age_branch)
    age_branch = Dense(256, activation='relu')(age_branch)
    age_output = Dense(age_classes, activation='softmax', name='age_output')(age_branch)
    
    # Gender prediction branch
    gender_branch = Dense(512, activation='relu')(x)
    gender_branch = Dropout(0.5)(gender_branch)
    gender_branch = Dense(256, activation='relu')(gender_branch)
    gender_output = Dense(gender_classes, activation='softmax', name='gender_output')(gender_branch)
    
    # Define model
    model = Model(inputs=base_model.input, outputs=[age_output, gender_output])
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss={'age_output': 'sparse_categorical_crossentropy',
                        'gender_output': 'sparse_categorical_crossentropy'},
                  metrics={'age_output': 'accuracy',
                           'gender_output': 'accuracy'})
    
    return model

def train_model(model, X_train, y_age_train, y_gender_train, X_val, y_age_val, y_gender_val, epochs=50, batch_size=32):
    history = model.fit(
        X_train,
        {'age_output': y_age_train, 'gender_output': y_gender_train},
        validation_data=(X_val, {'age_output': y_age_val, 'gender_output': y_gender_val}),
        epochs=epochs,
        batch_size=batch_size
    )
    
    return model, history