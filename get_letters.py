from io import BytesIO
from pathlib import Path

import cv2
import requests
import numpy as np
from PIL import Image

IMGS_PATH = Path.cwd() / "images"
IMGS_TRANS_PATH = IMGS_PATH / "trans"
IMGS_PATH.mkdir(exist_ok=True)
IMGS_TRANS_PATH.mkdir(exist_ok=True)

LEFT_PAD = 160
RIGHT_PAD = 130

BASE_PIC = "https://stardisk.xyz/projects/funpics/"

funpics_url = "https://stardisk.xyz/projects/funpics/funpics.php"

headers = {
    'authority': "stardisk.xyz",
    'accept': "*/*",
    'cache-control': "no-cache",
    'origin': "https://stardisk.xyz",
    'pragma': "no-cache",
    'referer': "https://stardisk.xyz/funpics/",
    'user-agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36"
}


def get_char_url(char: str) -> str:
    form_data = {"word": f"++{char}+"}
    response = requests.post(funpics_url, data=form_data, headers=headers)

    return BASE_PIC + response.text


def save_pic_from_url(url: str) -> Image:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    # The crop method from the Image module takes four coordinates as input.
    # The right can also be represented as (left+width)
    # and lower can be represented as (upper+height).
    img_w, img_h = img.size
    left = LEFT_PAD
    upper = 0
    right = img_w - RIGHT_PAD
    lower = img_h
    box = (left, upper, right, lower)

    cropped_img = img.crop(box)
    return cropped_img


def remove_bg_from_img(file_path: Path):
    img = cv2.imread(str(file_path))

    # add alpha channel!
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

    # threshold on white
    # Define lower and upper limits
    lower = np.array([220, 220, 220, 255])
    upper = np.array([255, 255, 255, 255])

    # Create mask to only select black
    thresh = cv2.inRange(img, lower, upper)

    # apply morphology
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # invert morp image
    mask = 255 - morph

    # apply mask to image
    img_result = cv2.bitwise_and(img, img, mask=mask)

    return img_result


def make_size_img(img):
    width, height, depth = img.shape
    max_side = max((width, height))
    new_alpha_squared_img = np.zeros((max_side, max_side, depth), img.dtype)
    margin_w = (max_side - width) // 2
    margin_h = (max_side - height) // 2

    new_alpha_squared_img[margin_w: margin_w + width, margin_h: margin_h + height] = img

    dsize = (100, 100)
    small_img = cv2.resize(new_alpha_squared_img, dsize)

    return small_img


def save_cropped_transparent_image_for_char(char: str):
    save_path = IMGS_PATH / f"letter_{char}.png"
    # --- this can be ignored once downloaded
    url = get_char_url(char)
    img = save_pic_from_url(url)
    img.save(save_path)
    print("saved", save_path)
    # ^^^

    processed_img = remove_bg_from_img(save_path)
    trans_save_path = IMGS_TRANS_PATH / f"letter_{char}_trans.png"
    resized_img = make_size_img(processed_img)
    cv2.imwrite(str(trans_save_path), resized_img)
    print("saved", trans_save_path)


def main():
    letters = (
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        "abcdefghijklmnoprqstuvwxyz"
        "1234567890"
        ".,!?:;()-"
    )
    for letter in letters:
        save_cropped_transparent_image_for_char(letter)


if __name__ == '__main__':
    main()
