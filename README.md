# Luxonis-OAK-D
Software for testing of Luxonis OAK-D product family

## Canny edge detection
For preprocessing purposes the contours will be extracted after applying the Canny algorithm to the image. This will produce a binary image that only shows the edges between the thresholds established. So far we've tested:
- Only Canny directly to RGB image: Rarely misses the target but produces tons of shapes in return


- Preprocessing (Grayscale + blur with 7x7 kernel) + Canny + Erosion + Dilation: Sometimes filters target hexagon but the image has less artifacts.
- Succesful testing with low canny threshold (< 50) and normal high canny threshold (300) - 25,241 very good results

## Spatial Location
To perform spatial location we generate a ROI based on the contours found that are of the shape targeted. To generate this ROI we use the centroid of the contour and the distance to the closest point. The reference used is the same from OpenCV images, (0,0) is top left corner. The generated ROI must be transformed to a depth ROI -> correlation based on resolution is not accurate. For now, transformation on resolution works as reference for future matching. Next step is to align RGB and Depth images to obtain the correct ROI for the depth and spatial location.

## Important Notes
[Spatial Location Calculator software](https://docs.luxonis.com/projects/api/en/latest/samples/SpatialDetection/spatial_location_calculator/#spatial-location-calculator) tested with OAK-D Pro without laser gives accurate measurements above 317 mm distance (only for 400P mono resolution, it increases with more resolution). Far distances accuracy is not clear but nonetheless out of scope for current applications.
    