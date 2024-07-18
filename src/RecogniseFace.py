import cv2
import numpy as np
from deepface import DeepFace as df # type: ignore

class RecogniseFace:
    """
    Class to detect faces and verify it against the reference image

    credit for detectFaces(): https://www.youtube.com/watch?v=i3sLv1sus0I
    credit for checkFace(): https://www.youtube.com/watch?v=pQvkoaevVMk

    """
    def __init__(self):
        self.faceMatch = False
        self.similarityScore = 0
        self.refFront = cv2.imread('photos/ref_Front.jpg')
        self.refSide = cv2.imread('photos/ref_Side.jpg')

        if self.refFront is None or self.refSide is None:
            raise ValueError("Reference image not found. Check the path to 'photos/reference.jpg'")

        self.frontFaceClassifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        self.sideFaceClassifier = cv2.CascadeClassifier('haarcascade_profileface.xml')

        if self.frontFaceClassifier.empty():
            raise ValueError("Failed to load Haar Cascade Front Face Classifier")

        if self.sideFaceClassifier.empty():
            raise ValueError("Failed to load Haar Cascade Face Profile classifer")        

    def detectFaces(self, img):
        """
        Function to detect faces

        :param img: UMat
            UMat data structure (Unified Memory Array).
            Its a multi-dimensional array that stores numbers as an image representation
        :return: UMat
            The input image with detected faces
        """
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frontFaces = self.frontFaceClassifier.detectMultiScale(grayImage, 1.3, 3) 
        print(frontFaces)
        sideFaces = self.sideFaceClassifier.detectMultiScale(grayImage, 1.3, 3)
        print(sideFaces)
        
        faces = []
        for value in frontFaces:
            faces.append(value)

        for value in sideFaces:
            faces.append(value)

        print(faces)
        
        if len(faces) == 0:
            return img

        #TODO: fix overlapping boxes. prioritise front face box
        if len(faces) > 4:
            for (x,y,w,h) in faces[0]:
                cv2.rectangle(img,(x,y),(x+w, y+h),(255,0,0), 2)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w, y+h),(255,0,0), 2)
        return img

    def checkFace(self, frame):
        """
        Function to verify face detected against self.referenceImage

        :param frame: UMat
        :return: boolean
            Returns True if a match is found
            Returns False otherwise
        """
        #TODO: figure out how to verify using multiple reference images
        #TODO: get side profile verified
        try:
            result = df.verify(frame, self.refFront, model_name='Facenet512', distance_metric="cosine") # returns a dictionary
            if result['verified']: # use value-pair of key 'verified' to return boolean value
                self.faceMatch = True
                self.similarityScore = self.calculateSimilarity(result['distance'])
                print('distance: ' + str(result['distance']) +'; similarity score: ' + str(self.similarityScore) +'%')
            else:
                self.faceMatch = False
        
        except ValueError as ve:
            print(f"ValueError during face verification: {ve}")
            self.faceMatch = False
        
        except Exception as e:
            print(f"Error during face verification: {e}")
            self.faceMatch = False
    
    def calculateSimilarity(self, distance):
        similarity = (1 - distance) * 100
        res = np.round(similarity, 2)
        return res

    def getFaceMatch(self):
        return self.faceMatch

    def getSimilarityScore(self):
        return self.similarityScore

    def getReferenceImage(self):
        return self.refFront, self.refSide
    
    def setReferenceImage(self, img, isFront):
        if isFront:
            self.refFront = img
        else:
            self.refSide = img
        return self