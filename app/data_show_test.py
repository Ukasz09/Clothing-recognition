from app.data_utils import *

if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = load_data()
    scale_data(train_images, test_images)
    display_first_images(train_images, train_labels)
