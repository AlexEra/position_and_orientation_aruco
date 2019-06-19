import cv2
import cv2.aruco as aruco
import pickle
from undistort import undist_img
from d_s_marker import detect_show_marker

cap = cv2.VideoCapture(0)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)  # 5X5_100 6X6_250
parameters = aruco.DetectorParameters_create()

# Load coeffs.
with open('cam_param.pkl', 'rb') as f:
    camera_param = pickle.load(f)
cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsInstrinsics, 
stdDeviationsExtrinsics = camera_param

while(True):
    # Capture frame-by-frame.
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s_img = img
    # Undistorting.
    img = undist_img(img, cameraMatrix, distCoeffs)
    # Show detected marker.
    detect_show_marker(s_img, gray, aruco_dict, parameters, cameraMatrix,
                       distCoeffs)
    # Press esc for close.
    if cv2.waitKey(5) == 27:
        break
cap.release()
cv2.destroyAllWindows()
