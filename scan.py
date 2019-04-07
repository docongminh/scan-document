# USAGE
# python scan.py --image images/page.jpg

from transform import four_point_transform
from skimage.filters import threshold_local
import argparse
import cv2
import imutils
import pytesseract
import matplotlib.pyplot as plt


def detect_edge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use Bilateral filter to remove noise while keeping edges sharp
    blur = cv2.bilateralFilter(gray, 5, 75, 75)
    edged = cv2.Canny(blur, 75, 200)
    return edged


def find_rect_contours(edged):
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for cnt in cnts:
        # approximate the contour
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            return approx

        # TODO: Try Convex Hull or Bounding Rectangle if approx contours failed
        # approx = cv2.convexHull(c)

        # rect = cv2.minAreaRect(c)
        # box = cv2.boxPoints(rect)
        # approx = np.int0(box)
        # cv2.drawContours(img,[approx],0,(0,0,255),2)
    return None


def apply_top_down_transform(image, contours, ratio):
    warped = four_point_transform(image, contours.reshape(4, 2) * ratio)
    return warped


def crop_then_transform_document(image):
    RESIZE_HEIGHT = 500
    orig = image.copy()
    ratio = image.shape[0] / float(RESIZE_HEIGHT)
    image = imutils.resize(image, height=RESIZE_HEIGHT)

    edged = detect_edge(image)
    contour = find_rect_contours(edged)
    if contour is None:
        h = orig.shape[0]
        w = orig.shape[1]
        contour = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)] 
        return contour, orig
    warped = apply_top_down_transform(orig, contour, ratio)
    contour = contour.reshape(4, 2).tolist()
    return contour, warped


def extract_text_from_image(image, lang='vie') -> str:
    # Convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = threshold_local(gray, 11, offset=10, method="gaussian")
    gray = (gray > T).astype("uint8") * 255

    config = '-l {lang}'.format(lang=lang)
    text = pytesseract.image_to_string(gray, config=config)
    lines = text.splitlines()
    text = '\n'.join(l.strip() for l in lines if l.strip())
    return text


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help='Path to the image to be scanned')
    ap.add_argument('-s', '--show', action='store_true', help='Show steps with images')
    args = ap.parse_args()

    # load the image and compute the ratio of the old height
    # to the new height, clone it, and resize it
    RESIZE_HEIGHT = 500
    image = cv2.imread(args.image)
    orig = image.copy()
    ratio = image.shape[0] / float(RESIZE_HEIGHT)
    image = imutils.resize(image, height=RESIZE_HEIGHT)

    print('STEP 1: Edge Detection')
    edged = detect_edge(image)

    print('STEP 2: Find contours of document')
    contour = find_rect_contours(edged)

    if contour is None:
        exit('Document not found')

    print('STEP 3: Apply top down transform')
    warped = apply_top_down_transform(orig, contour, ratio)

    print('STEP 4: Extract text from document')
    text = extract_text_from_image(warped, lang='vie')
    print(text)

    if args.show:
        plt.subplot(1, 4, 1, xticks=[], yticks=[], title='1. Original')
        plt.imshow(image, 'gray')

        plt.subplot(1, 4, 2, xticks=[], yticks=[], title='2. Edged')
        plt.imshow(edged, 'gray')

        plt.subplot(1, 4, 3, xticks=[], yticks=[], title='3. Outline')
        outline = image.copy()
        cv2.drawContours(outline, [contour], -1, (0, 255, 0), 2)
        plt.imshow(outline, 'gray')

        plt.subplot(1, 4, 4, xticks=[], yticks=[], title='4. Document')
        plt.imshow(warped, 'gray')

        plt.show()
