import os
import cv2


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def main():
    DATA_DIR = './data0'
    create_directory_if_not_exists(DATA_DIR)

    dataset_size = 100
    words = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
             'V', 'W', 'X', 'Y', 'Z']
    cap = cv2.VideoCapture(0)
    index = 3
    target_directory = os.path.join(DATA_DIR, str(words[index]))
    create_directory_if_not_exists(target_directory)

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(target_directory, '{}.jpg'.format(counter)), frame)
        counter += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
