import cv2
import subprocess
import numpy as np

def capture_from_video():
    # Start the process
    process = subprocess.Popen(
        ["./raspberrypi_video", "-tl", "3"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    
    # Loop to read frames
    try:
        while True:
            # Read the raw output from the process
            frame = process.stdout.read(640 * 480 * 2)  # Assuming 16-bit grayscale
            if not frame:
                break
            
            # Convert raw data to numpy array
            img = np.frombuffer(frame, dtype=np.uint16).reshape((480, 640))
            
            # Normalize to 8-bit for display
            cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
            img = np.uint8(img)
            
            # Resize for better display, if needed
            img_resized = cv2.resize(img, (640, 480))
            
            # Display the frame
            cv2.imshow("Thermal View", img_resized)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Terminating...")
    finally:
        # Cleanup
        process.terminate()
        cv2.destroyAllWindows()

# Call the function to start capturing
capture_from_video()
