import cv2 as cv
 
def readImage(path):
    """
    Function to read images
    :parameters:
    :path: string of image path
    :returns: image in the specified window
    """

    img = cv.imread(path)

    cv.imshow('Miku', img)

    cv.waitKey(0) # window is displayed for an infinite amount of time until any key input

# readImage('photos/miku.jpg')

def readVideo(path, isCamera):
    """
    Function to read videos
    :parameters:
    :path: string of video path
    :isCamera: boolean 
    :returns: video in the specified window
    """
    winName = ''
    if isCamera:
        capture = cv.VideoCapture(0) # enable laptop camera
        winName = 'Camera'
    else:
        capture = cv.VideoCapture(path) # use video
        winName = 'Music Video'

    while True:
        isTrue, frame = capture.read()
        cv.imshow(winName, frame)

        if cv.waitKey(20) & 0xFF==ord('d'): # break out if d is pressed
            break
    
    capture.release() # closes video file/capturing device
    cv.destroyAllWindows # closes windows

# readVideo('videos/love is war - 720p.mp4', False)
# readVideo('videos/love is war - 360p.mp4', False)
readVideo(None, True)