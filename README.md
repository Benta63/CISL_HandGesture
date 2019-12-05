# CISL_HandGesture
As part of an undergraduate research project with the Cognitive Immersive Systems Laboratory, I created (hopefully) a hand gesture recognition software

## Notes on Data

* Need to Split data into training/testing/validation
  * Training can be:
    * All inside data with gloves
    * Do a count. Maybe all with sleeves as well
  * Validation can be:
    * All inside without gloves
  * The results we will look at is:
    * All Outside data applied to NN

* Need to classify how well the algorithm does on certain types of images, e.g. Sleeves/No sleeves, no gloves vs gloves. Perhaps do some type of statistical tests?

## Python Packages Used

* numpy
 * 1.17.3
* pandas
 * 0.25.3
* pytorch
 * 1.0.1
* opencv
 * 3.4.2
* scikit-image
 * 0.15.0
* scikit-learn
 * 0.21.3
* torchvision
 * 0.2.2
* matplotlib
 * 3.1.1
* Maybe more. idk


