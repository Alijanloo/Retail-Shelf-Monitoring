import cv2

# 1. Initialize the video capture object
# Use camera index in production, or a file path/URL for testing
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # 2. Capture frame-by-frame
    # 'ret' (return value) is a boolean, 'frame' is the image array
    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # 3. Apply your processing (e.g., OpenCV, ML inference)
    # Example: Convert the frame to grayscale for simpler display
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 4. Display the resulting frame
    cv2.imshow("Live Camera Stream", gray_frame)

    # 5. Break the loop on 'q' key press
    if cv2.waitKey(1) == ord("q"):
        break

# 6. When everything is done, release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
