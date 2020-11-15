import numpy as np
import cv2
import copy
import os

Original_WIN = "Original video"
Heatmap_WIN = "Heatmapped video"

def main(input_file='vtest.avi', heatmap_color=cv2.COLORMAP_BONE, color_label="cv2.COLORMAP_BONE"):
    print ("Input source: " + input_file + " ", heatmap_color)

    cap = cv2.VideoCapture(input_file)
    #fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    #fgbg = cv2.createBackgroundSubtractorMOG2()
    fgbg = cv2.createBackgroundSubtractorKNN()
    
    # Debug parameters
    fps = cap.get(cv2.CAP_PROP_FPS) # Frames per second
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) # Frame count
    frame_number = 0 # Frame number

    success, frame = cap.read()
    first_frame = copy.deepcopy(frame)
    height, width = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).shape[:2]
    accum_image = np.zeros((height, width), np.uint8)
    while success and cap.isOpened():
        index = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        fgmask = fgbg.apply(gray)  # remove the background

        # for testing purposes, show the result of the background subtraction
        # cv2.imshow('diff-bkgnd-frame', fgmask)

        # apply a binary threshold only keeping pixels above thresh and setting the result to maxValue.  If you want
        # motion to be picked up more, increase the value of maxValue.  To pick up the least amount of motion over time, set maxValue = 1
        thresh = 2
        maxValue = 2
        ret, th1 = cv2.threshold(fgmask, thresh, maxValue, cv2.THRESH_BINARY)
        # for testing purposes, show the threshold image
        # cv2.imwrite('diff-th1.jpg', th1)

        # add to the accumulated image
        accum_image = cv2.add(accum_image, th1)
        # for testing purposes, show the accumulated image
        # cv2.imwrite('diff-accum.jpg', accum_image)

        # for testing purposes, control frame by frame
        # raw_input("press any key to continue")

        try:
            color_map_img = cv2.applyColorMap(accum_image, heatmap_color)
        except TypeError:
            print ("Could not apply color: " + heatmap_color)
            return
        im0 = cv2.resize(color_map_img, (1000, 980))
        font_size, font_thickness = 0.7, 2
        text = color_label[4:]
        x,y,w,h = 15,30,275,75
        # Draw black background rectangle
        cv2.rectangle(im0, (x, x), (x + w, y + h), (0,0,0), -1)
        im0 = cv2.putText(im0, text, (x + int(w/10),y + int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 55, 255), font_thickness)

        cv2.namedWindow(Original_WIN)
        cv2.moveWindow(Original_WIN, 1080, 0)
        cv2.namedWindow(Heatmap_WIN)
        cv2.moveWindow(Heatmap_WIN, 20, 0)
        cv2.imshow(Heatmap_WIN, im0)
        cv2.imshow(Original_WIN, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('k'):
            print("Terminated")
            #sys.exit(0)
        #print("Frame: ", frame_number)
        #print("FPS: ", fps)
        while frame_number < 1:
            os.system("xdotool getactivewindow windowmove -- -40 0")
            frame_number += 1
        frame_number += 1
        #cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()

    # apply a color map
    print("writing to file")
    color_image = im_color = cv2.applyColorMap(accum_image, heatmap_color)
    # for testing purposes, show the colorMap image
    # cv2.imwrite('diff-color.jpg', color_image)

    # overlay the color mapped image to the first frame
    result_overlay = cv2.addWeighted(first_frame, 0.7, color_image, 0.7, 0)

    # save the final overlay image
    cv2.imwrite('diff-overlay.jpg', result_overlay)

    # cleanup
    cap.release()
    cv2.destroyAllWindows()

color_maps = [cv2.COLORMAP_HSV, cv2.COLORMAP_PINK]
color_maps_labels = ["cv2.COLORMAP_HSV", "cv2.COLORMAP_PINK",]

if __name__=='__main__':
    import sys
    if len(sys.argv) < 2:
        main()
        print("\n\nUsed default video vtest.avi with single color\n\nFor all colors try: \n\n\tpython motion-heatmap.py ./vtest.avi all-colors\n\n")
    else:
        if (len(sys.argv) > 2 and 'all-colors' in str(sys.argv[2])):
            for i in range(len(color_maps)):
                print("Heatmap in " + str(color_maps_labels[i][4:]))
                main(input_file=sys.argv[1], heatmap_color=color_maps[i], color_label=color_maps_labels[i])
        else:
            main(input_file=sys.argv[1])