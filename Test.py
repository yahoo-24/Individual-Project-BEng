import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from UNet import preprocess_data, load_data
import matplotlib.pyplot as plt
import cv2, os
from sklearn.metrics import precision_score, recall_score, f1_score
#from MNV3 import dice_loss, bce_dice_loss

# Load model
from tensorflow.keras.models import load_model

# def dice_loss(y_true, y_pred, smooth=1e-6):
#     y_true_f = tf.reshape(y_true, [-1])
#     y_pred_f = tf.reshape(y_pred, [-1])
#     intersection = tf.reduce_sum(y_true_f * y_pred_f)
#     return 1 - (2.*intersection + smooth)/(tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# def bce_dice_loss(y_true, y_pred):
#     bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
#     return bce + dice_loss(y_true, y_pred)

# model = load_model("best_model_cpu.keras")  # use SavedModel folder
# model = load_model("mobilenetv3_unet_cpu.keras")  # use SavedModel folder

# Load and preprocess image
# img = load_img("//home//yahia//Downloads//HemoSet//pig3//imgs//002940.png", target_size=(224,224))
# img_array = img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)  # batch dimension
# img_array = preprocess_input(img_array)  # [-1,1] scaling

# Predict mask
# pred_mask = model.predict(img_array)[0,...,0]

# Visualize raw mask
# plt.imshow(pred_mask, cmap='gray')
# plt.colorbar()
# plt.show()

# # Apply threshold if desired
# binary_mask = (pred_mask > 0.5).astype(np.uint8)
# plt.imshow(binary_mask, cmap='gray')
# plt.show()

def evaluate_results(y_true, y_pred):
    # Calculate Precision
    precision = precision_score(y_true, y_pred, average='micro')

    # Calculate Recall
    recall = recall_score(y_true, y_pred, average='micro')

    # Calculate F1 Score
    f1 = f1_score(y_true, y_pred, average='micro')

    # Calculate IOU score
    intersect = y_true * y_pred
    union = y_true + y_pred
    union[union > 1] = 1
    iou = np.sum(intersect) / np.sum(union)

    return precision, recall, f1, iou


def preprocess_data_old(images, masks, img_size):
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
    return images_normalized, masks_normalized

def load_coloured_images():
    images = []
    for i in range(1, 11):
        img_folder = os.path.expanduser(f"~/Downloads/HemoSet/pig{i}/imgs")
        img_files = sorted(os.listdir(img_folder))
        for img_file in img_files:
            img_path = os.path.join(img_folder, img_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Add channel dimension
            images.append(img)
    return np.array(images)

if __name__ == "__main__":
    # model_1 = load_model("unet.keras")
    model_2 = load_model("unet_4.keras")
    print("Loading Data...")
    original_images, masks = load_data()
    coloured_images = load_coloured_images()
    print("Data Loaded!")
    # images_3channel, _ = preprocess_data(original_images, masks, (128, 128, 3))
    # images_3channel = cv2.cvtColor(images_3channel, cv2.COLOR_BGR2RGB)
    evaluate_masks = masks
    print("Processing Data...")
    images, masks = preprocess_data(original_images, masks, (128, 128))
    # images_1, masks_1 = preprocess_data_old(original_images, masks, (128, 128))
    print("Data Processed!")

    # loss, accuracy = model_1.evaluate(images_1, masks_1)
    # print(f"Loss: {loss}, Accuracy: {accuracy}")
    # loss, accuracy = model_2.evaluate(images, masks)
    # print(f"Loss: {loss}, Accuracy: {accuracy}")

    results_1 = []
    results_2 = []
    masks_to_test = []
    threshold = 0.85 # 0.49 for models 2,3,4

    # for i in range(len(images)):
    #     print(f"Test #{i}")
    #     sample_image = images[i:i+1]
    #     # sample_image_1 = images_1[i:i+1]
    #     # predicted_mask_1 = model_1.predict(sample_image_1)
    #     predicted_mask_2 = model_2.predict(sample_image)

    #     mask_rescaled = tf.squeeze(evaluate_masks[i]).numpy()
    #     masks_to_test.append(mask_rescaled)

    #     # output_rescaled_1 = tf.image.resize(predicted_mask_1, (480, 640))
    #     # output_rescaled_1 = tf.squeeze(output_rescaled_1).numpy()
    #     # output_rescaled_1[output_rescaled_1 >= 0.09] = 1
    #     # output_rescaled_1[output_rescaled_1 < 0.09] = 0
    #     # output_rescaled_1 = output_rescaled_1.astype(np.uint8)
    #     # results_1.append(output_rescaled_1)

    #     output_rescaled_2 = tf.image.resize(predicted_mask_2, (480, 640))
    #     output_rescaled_2 = tf.squeeze(output_rescaled_2).numpy()
    #     output_rescaled_2[output_rescaled_2 >= threshold] = 1
    #     output_rescaled_2[output_rescaled_2 < threshold] = 0
    #     output_rescaled_2 = output_rescaled_2.astype(np.uint8)
    #     results_2.append(output_rescaled_2)

    # masks_to_test = np.array(masks_to_test)
    # masks_to_test[masks_to_test > 0] = 1
    # masks_to_test = masks_to_test.astype(np.uint8)
    # # print(f"{type(masks_to_test)}, {type(masks_to_test[0])}, {masks_to_test.shape}")
    # # print(f"{type(np.array(results_1))}, {type(np.array(results_1)[0])}, {np.array(results_1).shape}")

    # # precision_1, recall_1, f1_1, iou_1 = evaluate_results(masks_to_test.flatten(), np.array(results_1).flatten())
    # precision_2, recall_2, f1_2, iou_2 = evaluate_results(masks_to_test.flatten(), np.array(results_2).flatten())

    # # print(f"Model 1: Precision: {precision_1}, Recall: {recall_1}, F1: {f1_1}, IOU: {iou_1}")
    # print(f"Model 2: Precision: {precision_2}, Recall: {recall_2}, F1: {f1_2}, IOU: {iou_2}")
    # Model 1: Precision: 0.9485561628335066, Recall: 0.9485561628335066, F1: 0.9485561628335066, IOU: 0.4209432139344042
    # Model 2: Precision: 0.960008848617247, Recall: 0.960008848617247, F1: 0.960008848617247, IOU: 0.5421138780500032
    # Model 3: Precision: 0.9429157225885741, Recall: 0.9429157225885741, F1: 0.9429157225885741, IOU: 0.20400471258119418
    # Model 4: Precision: 0.9771518517465783, Recall: 0.9771518517465783, F1: 0.9771518517465783, IOU: 0.7129498141889766
    # Model 5: Precision: 0.9738037447754244, Recall: 0.9738037447754244, F1: 0.9738037447754244, IOU: 0.6371219647921799

    while True:
        index = np.random.randint(0, len(images))

        sample_image = images[index:index+1]
        # sample_image_1 = images_1[index:index+1]
        # predicted_mask_1 = model_1.predict(sample_image_1)
        predicted_mask_2 = model_2.predict(sample_image)

        # Visualize input, ground truth, and prediction
        print(index)
        # plt.figure(figsize=(12, 4))
        # plt.subplot(1, 4, 1)
        # plt.title("Original Image")
        # plt.imshow(coloured_images[index])
        # plt.subplot(1, 4, 2)
        # plt.title("Input Image")
        # plt.imshow(sample_image[0].squeeze(), cmap='gray')
        # plt.subplot(1, 4, 3)
        # plt.title("Ground Truth")
        # plt.imshow(masks[index].squeeze(), cmap='gray')
        # plt.subplot(1, 4, 4)
        # plt.title("Predicted Mask")
        # plt.imshow(predicted_mask_1[0].squeeze(), cmap='gray')
        # plt.show()

        mask_rescaled = tf.image.resize(masks[index], (480, 640))
        mask_rescaled = tf.squeeze(mask_rescaled).numpy()

        # output_rescaled_1 = tf.image.resize(predicted_mask_1, (480, 640))
        # output_rescaled_1 = tf.squeeze(output_rescaled_1).numpy()
        # # print(output_rescaled_1)
        # output_rescaled_1[output_rescaled_1 >= 0.09] = 1
        # output_rescaled_1[output_rescaled_1 < 0.09] = 0
        # output_rescaled_1 = output_rescaled_1.astype(np.uint8)
        # out_1 = cv2.bitwise_or(coloured_images[index], coloured_images[index], mask=output_rescaled_1)

        output_rescaled_2 = tf.image.resize(predicted_mask_2, (480, 640))
        output_rescaled_2 = tf.squeeze(output_rescaled_2).numpy()
        # print(output_rescaled_2)
        output_rescaled_2[output_rescaled_2 >= 0.7] = 1
        output_rescaled_2[output_rescaled_2 < 0.7] = 0
        output_rescaled_2 = output_rescaled_2.astype(np.uint8)
        out_2 = cv2.bitwise_or(coloured_images[index], coloured_images[index], mask=output_rescaled_2)

        # plt.figure(figsize=(12, 4))
        # plt.subplot(1, 2, 1)
        # plt.title("Original Image")
        # plt.imshow(coloured_images[index])
        # plt.subplot(1, 2, 2)
        # plt.title("Masked Image")
        # plt.imshow(out)
        # plt.show()

        plt.figure(figsize=(8,6))

        plt.subplot(1, 2, 1)
        plt.title("Ground Truth")
        plt.imshow(coloured_images[index])
        plt.imshow(mask_rescaled, cmap="jet", alpha=0.5)  # transparent overlay
        plt.axis("off")

        # plt.subplot(1, 3, 2)
        # plt.title("Model 1")
        # plt.imshow(coloured_images[index])
        # plt.imshow(out_1, cmap="jet", alpha=0.5)  # transparent overlay
        # plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Model 2")
        plt.imshow(coloured_images[index])
        plt.imshow(out_2, cmap="jet", alpha=0.5)  # transparent overlay
        plt.axis("off")

        plt.show()
        
        cont = input("Continue? y/n: ")
        if cont.lower() == 'n':
            break