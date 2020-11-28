# Ball Detector (V8)

The aim of this project is to detect an arbitrary soccer ball for humanoid soccer robot team. The architecture of this detector consists of a search method, descriptor, and classifier.

## Search method
It is not sufficient to get ball candidates by using color and geometric features, since most of object in soccer field is white-colored and there exist another curve inside. I had developed a search method to overcome it, the algorithm is called Distance Weighting (DW). As the name suggests, it is an approximation of the Distance Transform algorithm, so basically the concept is similar, but the transformed image only contains information which proportional with how far a white pixel with a nearest black pixel. If your resource is adequate, I would like to recommend use actual Distance Transform algorithm instead.

## Descriptor
Nothing special in this part, I only use Local Binary Pattern.

## Classifier
Using Support Vector Machine.

## Note
* DW is not documented yet.
* Please use your own weight.
