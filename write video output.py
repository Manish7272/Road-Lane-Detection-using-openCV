import cv2
import numpy as np


def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def ROI(image):
    height = image.shape[0]
    triangle = np.array([[(0, 800), (1500, height), (1076, 554)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def display_line(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    if len(left_fit) > 0:
        left_fit_avg = np.average(left_fit, axis=0)
        left_line = coordinates(image, left_fit_avg)
    else:
        left_line = np.array([0, 0, 0, 0])  # Default if no left lines found

    if len(right_fit) > 0:
        right_fit_avg = np.average(right_fit, axis=0)
        right_line = coordinates(image, right_fit_avg)
    else:
        right_line = np.array([0, 0, 0, 0])  # Default if no right lines found

    return np.array([left_line, right_line])


# Open the video file
cap = cv2.VideoCapture("test.mp4")

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if no frame is returned

    canny_img = canny(frame)
    cropped_image = ROI(canny_img)

    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), 40, 5)
    if lines is not None:
        average_line = average_slope(frame, lines)
        line_image = display_line(frame, average_line)
        final_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    else:
        final_image = frame  # If no lines detected, keep original frame

    out.write(final_image)  # Write frame to output video
    cv2.imshow("Result", final_image)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
out.release()  # Release video writer
cv2.destroyAllWindows()
