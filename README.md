# Luxonis-OAK-D
Software for testing of Luxonis OAK-D product family

## Canny edge detection
For preprocessing purposes the contours will be extracted after applying the Canny algorithm to the image. This will produce a binary image that only shows the edges between the thresholds established. So far we've tested:
- Only Canny directly to RGB image: Rarely misses the target but produces tons of shapes in return


- Preprocessing (Grayscale + blur with 7x7 kernel) + Canny + Erosion + Dilation: Sometimes filters target hexagon but the image has less artifacts.
  - Succesful testing with low canny threshold (< 50) and normal high canny threshold (300)

## Important Notes
[Spatial Location Calculator software](https://docs.luxonis.com/projects/api/en/latest/samples/SpatialDetection/spatial_location_calculator/#spatial-location-calculator) tested with OAK-D Pro without laser gives accurate measurements above 317 mm distance. Far distances accuracy is not clear but nonetheless out of scope for current applications.
