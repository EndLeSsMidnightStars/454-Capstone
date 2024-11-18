import cv2
from pylepton import Lepton
import numpy as np

def capture_frame():
    with Lepton() as l:
        a, _ = l.capture()
        # Normalize the image to display
        cv2.normalize(a, a, 0, 65535, cv2.NORM_MINMAX)
        np.right_shift(a, 8, a)  # Fit to 8 bits
        return np.uint8(a)

while True:
    frame = capture_frame()
    # Resize for better display, optional
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Thermal View", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
