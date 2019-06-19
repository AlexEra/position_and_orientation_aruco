import cv2

def undist_img(img, cameraMatrix, distCoeffs):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, 
                                                      (w, h), 1,(w, h))
    # Undistort.
    dst = cv2.undistort(img, cameraMatrix, distCoeffs, None, newcameramtx)
    
    # Crop the image.
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst
