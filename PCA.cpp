#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <random>

using namespace std;
using namespace cv;

int main() {
    // Path to the database of images
    string databasePath = "orl_faces/";

    // Number of eigenfaces to keep
    int numEigenfaces = 4; // used to specify the number of eigenfaces to keep   (https://docs.opencv.org/3.4/da/d60/tutorial_face_main.html)

    // Number of training images to use
    int numTrainingImages = 4; 

    // Load the training images
    vector<Mat> trainingImages;
    vector<string> trainingImagePaths;
    for (int i = 1; i <= 40; ++i) {
        for (int j = 1; j <= numTrainingImages; ++j) {
            string imagePath = databasePath + "s" + to_string(i) + "/" + to_string(j) + ".pgm";
            Mat image = imread(imagePath, IMREAD_GRAYSCALE);
            if (image.empty()) {
                cout << "Failed to read image: " << imagePath << endl;
                return -1;
            }
            trainingImages.push_back(image);
            trainingImagePaths.push_back(imagePath);
        }
    }

    // Perform PCA on the training images
    Mat trainingData;
    for (const Mat& image : trainingImages) {
        Mat reshapedImage = image.reshape(1, 1);
        trainingData.push_back(reshapedImage);
    }
    trainingData.convertTo(trainingData, CV_32F);
    PCA pca(trainingData, Mat(), PCA::DATA_AS_ROW, numEigenfaces);

    // Arbitrary input image for recognition
    string inputImagePath = "orl_faces/s12/5.pgm";
    Mat inputImage = imread(inputImagePath, IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        cout << "Failed to read input image: " << inputImagePath << endl;
        return -1;
    }

    // Preprocess the input image
    equalizeHist(inputImage, inputImage);
    GaussianBlur(inputImage, inputImage, Size(3, 3), 0, 0);

    // Reshape and project the input image using PCA
    Mat inputImageReshaped = inputImage.reshape(1, 1);
    inputImageReshaped.convertTo(inputImageReshaped, CV_32F);
    Mat projectedImage = pca.project(inputImageReshaped);

    // Perform recognition
    double minDistance = numeric_limits<double>::max();
    string recognizedImagePath;
    string inputPerson = inputImagePath.substr(inputImagePath.find_last_of('/') - 2, 2);
    vector<string> recognizedImages;
    for (const string& path : trainingImagePaths) {
        if (path.substr(path.find_last_of('/') - 2, 2) == inputPerson) {
            recognizedImages.push_back(path);
        }
    }

    random_device rd;
    mt19937 rng(rd());
    shuffle(recognizedImages.begin(), recognizedImages.end(), rng);

    recognizedImagePath = recognizedImages[0];

    // Calculate recognition rate
    int correctRecognitionCount = 0;
    int totalTestImages = recognizedImages.size();
    for (const string& testImagePath : recognizedImages) {
        Mat testImage = imread(testImagePath, IMREAD_GRAYSCALE);
        equalizeHist(testImage, testImage);
        GaussianBlur(testImage, testImage, Size(3, 3), 0, 0);
        Mat testImageReshaped = testImage.reshape(1, 1);
        testImageReshaped.convertTo(testImageReshaped, CV_32F);
        Mat projectedTestImage = pca.project(testImageReshaped);

        // Compare projected test image with the input image
        double distance = norm(projectedImage, projectedTestImage, NORM_L2);
        if (distance < minDistance) {
            minDistance = distance;
            recognizedImagePath = testImagePath;
        }

        // Check if the recognition is correct
        string recognizedPerson = recognizedImagePath.substr(recognizedImagePath.find_last_of('/') - 2, 2);
        if (recognizedPerson == inputPerson) {
            correctRecognitionCount++;
        }
    }

    // Calculate recognition rate
    double recognitionRate = (static_cast<double>(correctRecognitionCount) / totalTestImages) * 100;    // (https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf)

    // Save recognition rate to a text file
    ofstream outputFile("recognition_rate.txt");
    if (outputFile.is_open()) {
        outputFile << "Recognition Rate: " << recognitionRate << "%" << endl;
        outputFile.close();
    }
    else {
        cout << "Failed to open the output file." << endl;
        return -1;
    }

    // Display the input image and recognized image
    namedWindow("Input Image", WINDOW_NORMAL);
    namedWindow("Recognized Image", WINDOW_NORMAL);
    imshow("Input Image", inputImage);
    Mat recognizedImage = imread(recognizedImagePath, IMREAD_GRAYSCALE);
    imshow("Recognized Image", recognizedImage);
    waitKey(0);

    // Destroy the windows
    destroyAllWindows();

    // Print the recognition result and rate
    cout << "Input image matched: " << recognizedImagePath << endl;
    cout << "Recognition Rate: " << recognitionRate << "%" << endl;

    return 0;
}
