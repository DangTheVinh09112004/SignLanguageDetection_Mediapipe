import cv2
import os
import shutil


def create_directory(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)


def capture_images(category, num_samples):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capture image for sign language detection")
    image_counter = 0
    create_directory("My_Data/{}".format(category))
    print("Collecting data for {}".format(category))
    while image_counter < num_samples:
        flag, frame = cam.read()
        if not flag:
            break
        cv2.imshow("Capture sign for language data", frame)
        k = cv2.waitKey(1)

        if k % 256 == 27:
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            img_name = "My_Data/{}/image_{}.jpg".format(category, image_counter)
            cv2.imwrite(img_name, frame)
            image_counter += 1
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    categories = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L"]
    num_samples = 100
    for category in categories:
        capture_images(category, num_samples)
