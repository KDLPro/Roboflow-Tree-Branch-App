from tkinter import Tk, filedialog
import cv2
import sys


# Opening Video from file
root = Tk()
root.withdraw()
root.filename = filedialog.askopenfilename(
                    initialdir = "videos", 
                    filetypes=(("MP4 Video files", "*.mp4"), 
                               ("FLV video files", "*.flv")))
video = root.filename

if not video:
    print("No valid input video.") 
    sys.exit()


# Playing Video from file
if video:  
    cap = cv2.VideoCapture(video)    
    _, frame = cap.read()     

    frame_number = 0  

    while frame is not None:
        cv2.imshow('Image', frame) 
        _, frame = cap.read()   # Read the next frame
        

        image_path = 'data/images/'
        image_name = 'frame-{}.png'.format(frame_number)
        frame_number = frame_number + 1
        cv2.imwrite(image_path+image_name, frame)  # Saves the frame in the directory

        if cv2.waitKey(1) & 0xFF == 27: 
            break # Exit successfully if 'ESC' button is pressed.

    cap.release()
    cv2.destroyAllWindows()
