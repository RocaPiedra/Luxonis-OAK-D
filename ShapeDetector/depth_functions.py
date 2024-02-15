import depthai
import cv2

def paintSpatialData(depthData, depthFrameColor, color = (255, 255, 255), fontType = cv2.FONT_HERSHEY_TRIPLEX):
    roi = depthData.config.roi
    roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
    xmin = int(roi.topLeft().x)
    ymin = int(roi.topLeft().y)
    xmax = int(roi.bottomRight().x)
    ymax = int(roi.bottomRight().y)

    depthMin = depthData.depthMin
    depthMax = depthData.depthMax

    fontType = cv2.FONT_HERSHEY_TRIPLEX
    cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, 1)
    cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), fontType, 0.5, color)
    cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), fontType, 0.5, color)
    cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), fontType, 0.5, color)