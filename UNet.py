import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Input, concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def preprocess_data(images, masks, img_size):
    """
    Resizes and normalizes images and masks.
    
    Args:
        images (numpy array): Input images.
        masks (numpy array): Ground truth masks.
        img_size (tuple): Desired image size (height, width).
    
    Returns:
        images, masks: Preprocessed images and masks.
    """
    images_resized = [tf.image.resize(image, img_size) for image in images]
    masks_resized = [tf.image.resize(mask, img_size) for mask in masks]
    
    images_normalized = np.array(images_resized) / 255.0
    masks_normalized = np.array(masks_resized) / 255.0
    masks_normalized[masks_normalized > 0] = 1
    return images_normalized, masks_normalized


def unet_model(input_shape):
    """
    Defines a basic U-Net architecture for image segmentation.
    
    Args:
        input_shape (tuple): Shape of the input image (height, width, channels).
    
    Returns:
        model (tf.keras.Model): Compiled U-Net model.
    """
    inputs = Input(shape=input_shape)
    # Encoder: Downsampling path
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    # Bottleneck
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    # Decoder: Upsampling path
    u1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
    u1 = concatenate([u1, c2])
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    u2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u2 = concatenate([u2, c1])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(c5)
    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_data():
    """
    Loads the images and masks from the folder.
    
    Returns:
        images (np.array): Array of images in grayscale.
        masks (np.array): Array of masks in grayscale.
    """
    images = []
    masks = []
    for i in range(1, 11):
        img_folder = os.path.expanduser(f"~/Downloads/HemoSet/pig{i}/imgs")
        mask_folder = os.path.expanduser(f"~/Downloads/HemoSet/pig{i}/labels")
        img_files = sorted(os.listdir(img_folder))
        mask_files = sorted(os.listdir(mask_folder))
        for img_file, mask_file in zip(img_files, mask_files):
            img_path = os.path.join(img_folder, img_file)
            mask_path = os.path.join(mask_folder, mask_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # Add channel dimension
            img = img[..., np.newaxis]
            mask = mask[..., np.newaxis]
            images.append(img)
            masks.append(mask)
    return np.array(images), np.array(masks)

if __name__ == "__main__":
    # Initialize the U-Net model
    model = unet_model(input_shape=(128, 128, 1))
    # Train the model
    images_original, masks = load_data()
    images, masks = preprocess_data(images_original, masks, (128, 128))
    # early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)
    history = model.fit(images, masks, epochs=20, batch_size=16, validation_split=0.2, callbacks=[checkpoint])

    loss, accuracy = model.evaluate(images, masks)
    print(f"Loss: {loss}, Accuracy: {accuracy}")
    model.save("unet_5.keras")
    model.save("u_net_5.h5")

    # Predict on a sample
    sample_image = images[0:1]
    predicted_mask = model.predict(sample_image)
    # Visualize input, ground truth, and prediction
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(images_original[0])
    plt.subplot(1, 4, 2)
    plt.title("Input Image")
    plt.imshow(sample_image[0].squeeze(), cmap='gray')
    plt.subplot(1, 4, 3)
    plt.title("Ground Truth")
    plt.imshow(masks[0].squeeze(), cmap='gray')
    plt.subplot(1, 4, 4)
    plt.title("Predicted Mask")
    plt.imshow(predicted_mask[0].squeeze(), cmap='gray')
    plt.show()

    # More predicitons
    while True:
        index = np.random.randint(0, len(images))

        sample_image = images[index:index+1]
        predicted_mask = model.predict(sample_image)

        # Visualize input, ground truth, and prediction
        print(index)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 4, 1)
        plt.title("Original Image")
        plt.imshow(images_original[index])
        plt.subplot(1, 4, 2)
        plt.title("Input Image")
        plt.imshow(sample_image[0].squeeze(), cmap='gray')
        plt.subplot(1, 4, 3)
        plt.title("Ground Truth")
        plt.imshow(masks[index].squeeze(), cmap='gray')
        plt.subplot(1, 4, 4)
        plt.title("Predicted Mask")
        plt.imshow(predicted_mask[0].squeeze(), cmap='gray')
        plt.show()
        
        cont = input("Continue? y/n")
        if cont.lower() == 'n':
            break