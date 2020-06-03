from app.data_utils import *

if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = load_data()
    train_images_scaled, test_images_scaled = scale_data(train_images, test_images)
    display_first_images(train_images_scaled, train_labels, qty=16)
    display_first_images(train_images_scaled, train_labels, color_map=None, qty=10)
    display_first_images(train_images_scaled, train_labels, color_map="inferno", qty=10)
    exit(0)
