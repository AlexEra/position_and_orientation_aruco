import cv2
import cv2.aruco as aruco
import numpy as np
import math
import transforms3d


def detect_show_marker(img, gray, aruco_dict, parameters, cameraMatrix,
                       distCoeffs):
    detected_1, detected_2 = False, False
    i, j = None, None
    distance_1, distance_2 = None, None
    font = cv2.FONT_HERSHEY_SIMPLEX
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict,
                                                          parameters=parameters)
    img = aruco.drawDetectedMarkers(img, corners, ids)
    if ids is not None:
        i = 6  # Id of aruco - reference system.
        j = 5  # Id of target aruco.
        for k in range(0,len(ids)):
            rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[k],
                                                                       0.045,
                                                                       cameraMatrix,
                                                                       distCoeffs)
            if ids[k] == i:
                img = aruco.drawAxis(img, cameraMatrix, distCoeffs, rvec, tvec, 0.05)
                c_rvec = rvec
                c_tvec = tvec
                distance_1 = tvec[0][0][2]
                detected_1 = True
            elif ids[k] == j:
                n_rvec = rvec
                n_tvec = tvec
                distance_2 = tvec[0][0][2]
                detected_2 = True
            if (detected_1 == True) and (detected_2 == True):
                frvec, ftvec = relatPos(c_rvec, c_tvec, n_rvec, n_tvec)
                """ frvec - orientation vector of the marker regarding the reference 
                system.
                ftvec -  position of aruco regarding the rs.
                n_tvec - position of aruco regarding camera. """
                
                # Orientation aruco regarding the rs.
                angles0 = rotmtx_to_euler_angles(cv2.Rodrigues(frvec)[0]) 
                
                n_rmat = cv2.Rodrigues(n_rvec)[0]
                
                # Orientation aruco regarding camera.
                angles1 = rotmtx_to_euler_angles(n_rmat)
                
                # Camera position regarding aruco.
                pos_cam_to_aruco = -np.matrix(n_rmat).T * np.matrix(n_tvec).T
             
                cam_rotmtx =  np.matrix(n_rmat).T
                
                # Reverse orientation camera to the aruco.
                angles2 = rotmtx_to_euler_angles(cam_rotmtx) 
                
                rmat = cv2.Rodrigues(c_rvec)[0]
                
                # Camera position regarding the rs.
                pos_cam_to_rs = -np.matrix(rmat).T * np.matrix(c_tvec).T
                
                # Reverse orientation camera regarding the rs.
                angles3 = rotmtx_to_euler_angles(rmat)
    if (distance_1 is not None):
        cv2.putText(img, 'Id' + str(i) + ' %.2fsm' % (distance_1*100), (0, 64), font, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
    if (distance_2 is not None):
        cv2.putText(img, 'Id' + str(j) + ' %.2fsm' % (distance_2*100), (0, 104), font, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
    return cv2.imshow('frame', img)  # Final img.


def rotmtx_to_euler_angles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6 
    if  not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else :
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def relatPos(rvec1, tvec1, rvec2, tvec2):
    rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape((3, 1))
    rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1))
    irvec, itvec = inverse_vec(rvec2, tvec2)
    mtx = cv2.composeRT(rvec1, tvec1, irvec, itvec)
    composedRvec, composedTvec = mtx[0], mtx[1]
    composedRvec = composedRvec.reshape((3, 1))
    composedTvec = composedTvec.reshape((3, 1))
    return composedRvec, composedTvec


def inverse_vec(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    R = np.matrix(R).T
    itvec = np.dot(R, np.matrix(-tvec))
    irvec, _ = cv2.Rodrigues(R)
    return irvec, itvec
