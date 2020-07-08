#Self-Driving Car Engineer Nanodegree
#Project: Finding Lane Lines on the Road
#In this project, you will use the tools you learned about in the lesson to
#identify lane lines on the road. You can develop your pipeline on a series of
#individual images, and later apply the result to a video stream

#Import the libraries to be used in the project
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    plt.imshow(mask, cmap='gray')
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    return lines, line_img

def draw_test(img, lines, color=[255, 0, 0], thickness=10):
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img,(x1,y1),(x2,y2),color,thickness)
        
    
def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    
    l_added_slope = 0
    l_added_Y_intercept = 0
    l_counter = 0
    l_slope = 0
    l_Y_intercept = 0
    
    r_added_slope = 0
    r_added_Y_intercept = 0
    r_counter = 0
    r_slope = 0
    r_Y_intercept = 0

    for line in lines:
        for x1,y1,x2,y2 in line:        
            # Slope of each line using the formula (y2-y1) = m(x2-x1)
            m = (y2-y1)/(x2-x1)
            b = y2 - x2*m
            if(m == 'inf'):
                print("PROBLEMA AQUI")
            
            # Left lane
            elif (m<0):
                """if(x1<x1_left):
                    x1_left = x1
                if(x2>x2_left):
                    x2_left = x2
                if(y1>y1_left):
                    y1_left = y1
                if(y2<y2_left):
                    y2_left = y2"""    
                l_counter += 1
                l_added_slope += m
                l_added_Y_intercept += b
                    
            #Right Lane        
            else:
                """if(x1<x1_right):
                    x1_right = x1
                if(x2>x2_right):
                    x2_right = x2
                if(y1<y1_right):
                    y1_right = y1
                if(y2>y2_right):
                    y2_right = y2"""
                r_counter += 1
                r_added_slope += m
                r_added_Y_intercept += b
    
    #### Got an issue where in new iterations the program found no lines and since
    #### was the first iteration the value of last_known_slope was random, or the
    #### assigned value. For this, its better to mantain the recorded value over 
    #### all the iterations during the life-time of the program
    global last_known_l_slope
    global last_known_l_Y_intercept
    global last_known_r_slope
    global last_known_r_Y_intercept
    
    #Now we need to find the average slope and Y intercept for each lane
    #We have to make sure there is no Division by 0, meaning at least one line
    #was detected in the image counter must be different than 0
    
    if(l_counter != 0):
        l_slope = l_added_slope/l_counter
        if(l_slope == 'inf'):                  #Getting a infiniry value problem
            l_slope = last_known_l_slope
        l_Y_intercept = l_added_Y_intercept/l_counter
        last_known_l_slope = l_slope
        last_known_l_Y_intercept = l_Y_intercept
    #In In case no left lines are detected, we still need to draw a line for the
    #car to follow, so let's use the last known slope to draw the line
    else:
        l_slope = last_known_l_slope
        #l_Y_intercept = last_known_l_Y_intercept
        
    #Same for the right lane
    if(r_counter != 0):
        r_slope = r_added_slope/r_counter
        if(r_slope == 'inf'):                  #Getting a infiniry value problem
            r_slope = last_known_l_slope
        r_Y_intercept = r_added_Y_intercept/r_counter
        last_known_r_slope = r_slope
        last_known_r_Y_intercept = r_Y_intercept
    else:
        r_slope = last_known_r_slope
        #r_Y_intercept = last_known_r_Y_intercept
        
    #With the average slope and Y intercept, we can use the line formula
    # Y = m*X + b to calculate an average line of all the lines found.
    
    #Firstly we DO know the Y values for our points of interest. The Y value
    #at the bottom which is the maximum value, and the Y value we used for our
    #Region of interest for the mask.
    imshape = img.shape
    Y_bottom = imshape[0]
    Y_top = imshape[0]/1.6
    
    #For X value X = (Y - b)/m
    X_l_bottom = (Y_bottom - l_Y_intercept)/l_slope
    X_l_top = (Y_top - l_Y_intercept)/l_slope
    X_r_bottom = (Y_bottom - r_Y_intercept)/r_slope
    X_r_top = (Y_top - r_Y_intercept)/r_slope
    
    #With all the points, we need to draw both left and right lines.
    #Since cv2.line accept only integers (pixels are always integers) we need
    #to convert the X and Y values to integers
    cv2.line(img,(int(round(X_l_bottom)), int(round(Y_bottom))),
                 (int(round(X_l_top)),int(round(Y_top))),color,thickness)
    cv2.line(img,(int(round(X_r_bottom)), int(round(Y_bottom))),
                 (int(round(X_r_top)),int(round(Y_top))),color,thickness)
    plt.imshow(img, cmap='gray')
            
###############################################################################
###################### P R O G R A M   S T A R T ##############################
###############################################################################

def lane_detection(img, kernel_size=7,low_threshold=50,high_threshold=120,rho=2,
                   theta=np.pi/180,threshold=15, min_line_len=30, max_line_gap=15):
    color_image = np.copy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #grayscale conversion
    plt.imshow(gray, cmap='gray')
    
    #Apply Gaussian smoothing
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    plt.imshow(blur_gray, cmap='gray')
    
    # Define our parameters for Canny and apply
    edges = canny(blur_gray, low_threshold, high_threshold)
    plt.imshow(edges, cmap='gray')
    
    #Only see the region of our interest
    #First, lets define which is our region of interest with a four sided polygon
    imshape = img.shape
    vertices = np.array([[(imshape[1]/20, imshape[0]),
                          (imshape[1]/2.20, imshape[0]/1.6),
                          (imshape[1]/1.80, imshape[0]/1.6),
                          (imshape[1]/1.005,imshape[0])]],
                        dtype=np.int32)
    
    #Define in our Canny image the region of interest
    masked_edges = region_of_interest(edges, vertices)
    plt.imshow(masked_edges, cmap='gray')
    # Use the Hough transform to detect and Highlight lines
    lines, line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    plt.imshow(line_image, cmap='gray')
    #draw_test(color_image, lines)
    draw_lines(line_image, lines)
    color_image = cv2.addWeighted(color_image, 0.8, line_image, 0.4, 0) 
    plt.imshow(color_image)
    return color_image
   

    
### Define importan features for lane detection
kernel_size = 7;  #Odd number for Gaussian Smoothing Kernel
# Define the Hough transform parameters
rho = 2 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15    # minimum number of votes (intersections in Hough grid cell)
min_line_length = 30 #minimum number of pixels making up a line
max_line_gap = 3  # maximum gap in pixels between connectable line segments

### Using the Lane detection function for each image
img = mpimg.imread('test_images/solidWhiteCurve.jpg')
plt.imshow(img)
lanes1 = lane_detection(img, kernel_size = kernel_size, threshold = 5,rho = rho, 
                        theta=theta, min_line_len=min_line_length, max_line_gap=max_line_gap)
mpimg.imsave("test_images/solidWhiteCurve_lanedetection.png", lanes1)


image2 = mpimg.imread('test_images/solidWhiteRight.jpg')
plt.imshow(image2)
lanes2 = lane_detection(image2, kernel_size = kernel_size, threshold = 5,rho = rho, 
                        theta=theta, min_line_len=min_line_length, max_line_gap=max_line_gap)
mpimg.imsave("test_images/solidWhiteRight_lanedetection.png", lanes2)


image3 = mpimg.imread('test_images/solidYellowCurve.jpg')
plt.imshow(image3)
lanes3 = lane_detection(image3, kernel_size = kernel_size, threshold = 5,rho = rho, 
                        theta=theta, min_line_len=min_line_length, max_line_gap=max_line_gap)
mpimg.imsave("test_images/solidYellowCurve_lanedetection.png", lanes3)


image4 = mpimg.imread('test_images/solidYellowCurve2.jpg')
plt.imshow(image4)
lanes4 = lane_detection(image4, kernel_size = kernel_size, threshold = 5,rho = rho, 
                        theta=theta, min_line_len=min_line_length, max_line_gap=max_line_gap)
mpimg.imsave("test_images/solidYellowCurve2_lanedetection.png", lanes4)


image5 = mpimg.imread('test_images/solidYellowLeft.jpg')
plt.imshow(image5)
lanes5 = lane_detection(image5, kernel_size = kernel_size, threshold = 5,rho = rho, 
                        theta=theta, min_line_len=min_line_length, max_line_gap=max_line_gap)
mpimg.imsave("test_images/solidYellowLeft_lanedetection.png", lanes5)


image6 = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')
plt.imshow(image6)
lanes6 = lane_detection(image6, kernel_size = kernel_size, threshold = 5,rho = rho, 
                        theta=theta, min_line_len=min_line_length, max_line_gap=max_line_gap)
mpimg.imsave("test_images/whiteCarLaneSwitch_lanedetection.png", lanes6)



################# Using the Lane detection function for videos #################

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

#white_output = 'test_videos_output/solidWhiteRight_raw.mp4'
white_output = 'test_videos_output/solidWhiteRight.mp4'
#yellow_output = 'test_videos_output/solidYellowLeft_raw.mp4'
yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
clip2 = VideoFileClip("test_videos/solidYellowLeft.mp4")
white_clip = clip1.fl_image(lane_detection) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
yellow_clip = clip2.fl_image(lane_detection) #NOTE: this function expects color images!!
yellow_clip.write_videofile(yellow_output, audio=False)


############################## C H A L L E N G E ##############################
challenge_output = 'test_videos_output/challenge.mp4'
clip3 = VideoFileClip("test_videos/challenge.mp4")
challenge_clip = clip3.fl_image(lane_detection) #NOTE: this function expects color images!!
challenge_clip.write_videofile(challenge_output, audio=False)
