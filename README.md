# openCV-tracker
## Physical Components
- raspberry pi
- arduino
- any swivel components
- HDMI charging cable

# OpenCV documentation
## On the VideoCapture class
https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html#a85b55cf6a4a50451367ba96b65218ba1

## On the CascadeClassifier class
https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

# Setting up your venv
pip install opencv-python deepface
a pop-up will show asking to create a virtual environment - click Yes

# Activating venv
python -m venv venv
cd openCV-tracker
source venv/bin/activate (MacOS) or venv\Scripts\activate (Windows)
pip install -r requirements.txt

# Running the program
Make sure you're in the root directory (C:\~\openCV-tracker)
Then run the command 'python src/Main.py'