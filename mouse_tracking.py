import cv2
import time
import autopy
import shutil
import numpy as np
import hand_tracking as htm
from pathlib import Path


def delete_cache(input_path):
    for path in input_path.iterdir():
        if path.is_dir() and path.name == '__pycache__':
            shutil.rmtree(path)
        elif path.name == 'tempCodeRunnerFile.py':
            path.unlink()
        elif path.is_dir():
            delete_cache(path)


def create_frame_rate(img, prev_time):
    cur_time = time.time()
    fps = 1 / (cur_time - prev_time)
    cv2.putText(img, str(int(fps)), (20, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    return cur_time


def get_index_fingers(lm_list):
    try:
        return lm_list[8][1:]
    except IndexError:
        return None, None


def check_moving_mode(fingers_up):
    if fingers_up[1] == 1 and fingers_up[2] == 1:
        return False
    return True


def check_click_mode(fingers_up):
    if fingers_up[1] == 1 and fingers_up[2] == 1:
        return True
    return False


def interpolate_coords(x, y, frame_rate, cam_width, cam_height, screen_width, screen_height):
    interpolated_x = np.interp(
        x, (frame_rate, cam_width - frame_rate), (0, screen_width))
    interpolated_y = np.interp(
        y, (frame_rate, cam_height - 3 * frame_rate), (0, screen_height))

    return interpolated_x, interpolated_y


def smoothen_values(smooth_rate, x, y, prev_x, prev_y):
    cur_x = prev_x + (x - prev_x) / smooth_rate
    cur_y = prev_y + (y - prev_y) / smooth_rate

    return cur_x, cur_y


def main():
    dir_path = Path(__file__).parent

    screen_width, screen_height = autopy.screen.size()
    cam_width, cam_height = 1024, 768
    frame_rate = 100

    prev_time = 0
    prev_x, prev_y = 0, 0
    cur_x, cur_y = 0, 0
    smooth_rate = 3

    cap = cv2.VideoCapture(0)
    cap.set(3, cam_width)
    cap.set(4, cam_height)

    hand_detector = htm.HandDetector(maxHands=1)

    while True:
        _, img = cap.read()

        img = cv2.flip(img, 180)
        img = cv2.resize(img, (cam_width, cam_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = hand_detector.find_hands(img=img)

        cv2.rectangle(img, (frame_rate, frame_rate), (cam_width -
                      frame_rate, cam_height - frame_rate), (255, 0, 255), 2)

        lm_list, _ = hand_detector.find_position(img=img)
        ind_fing_x, ind_fing_y = get_index_fingers(lm_list)

        if cv2.waitKey(1) == ord(' '):
            break

        try:
            fingers_up = hand_detector.fingers_up()
            if check_moving_mode(fingers_up):
                interpolated_x, interpolated_y = interpolate_coords(x=ind_fing_x, y=ind_fing_y, frame_rate=frame_rate,
                                                                    cam_width=cam_width, cam_height=cam_height,
                                                                    screen_width=screen_width, screen_height=screen_height)
                cur_x, cur_y = smoothen_values(smooth_rate=smooth_rate, x=interpolated_x,
                                               y=interpolated_y, prev_x=prev_x, prev_y=prev_y)
                autopy.mouse.move(cur_x, cur_y)
                cv2.circle(img, (ind_fing_x, ind_fing_y),
                           15, (255, 0, 255), cv2.FILLED)
                prev_x, prev_y = cur_x, cur_y

            if check_click_mode(fingers_up):
                distance, img, line_info = hand_detector.find_distance(
                    8, 12, img)
                if distance < 45:
                    cv2.circle(
                        img, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)
                    autopy.mouse.click()
                    time.sleep(0.2)

            prev_time = create_frame_rate(img=img, prev_time=prev_time)
            cv2.imshow('AI mouse', img)
        except IndexError:
            prev_time = create_frame_rate(img=img, prev_time=prev_time)
            cv2.imshow('AI mouse', img)
            continue

    cap.release()
    cv2.destroyAllWindows()
    delete_cache(dir_path)


if __name__ == '__main__':
    main()
