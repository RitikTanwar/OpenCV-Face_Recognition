import cv2

img = cv2.imread(
    'Scrapy/amazon/amazon_data/watch/Vills Laurrens Pack of 4 Multicolour Analog Analog Watch for Men and Boy/Main_image.jpg')

cv2.imshow('Watch image', img)
gray = cv2.imread(
    'Scrapy/amazon/amazon_data/watch/Vills Laurrens Pack of 4 Multicolour Analog Analog Watch for Men and Boy/Main_image.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Gray Watch image', gray)
cv2.waitKey(0)  # Infinite time
# cv2.waitKey(250)  # Disappeared after 250ms
cv2.destroyAllWindows()
