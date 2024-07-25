#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

// Avoiding name mangling
extern "C" {

    // Attributes to prevent 'unused' function from being removed and to make it visible
    __attribute__((visibility("default"))) __attribute__((used))
    void get_perspective_transform(double *srcPoints, double *dstPoints, double *perspectiveMatrix) {
        Point2f src[4];
        Point2f dst[4];

        // Fill in the source and destination points using the point tables provided.
        for (int i = 0; i < 4; i++) {
            src[i] = Point2f(srcPoints[2 * i], srcPoints[2 * i + 1]);
            dst[i] = Point2f(dstPoints[2 * i], dstPoints[2 * i + 1]);
        }

        // Calculate the perspective transformation matrix
        Mat transformMatrix = getPerspectiveTransform(src, dst);

        // Copy the transformation matrix into the table provided
        std::memcpy(perspectiveMatrix, transformMatrix.ptr<double>(),
                    transformMatrix.total() * transformMatrix.elemSize());
    }



    // Function to apply perspective transformation and return image data
    __attribute__((visibility("default"))) __attribute__((used))
    void warpPerspectiveAndGetBuffer(
            uchar *imageData,
            int imageWidth,
            int imageHeight,
            double *perspectiveMatrix,
            int outputWidth,
            int outputHeight,
            uchar *outputBuffer // Buffer to store the transformed image
    ) {

        Mat input(imageHeight, imageWidth, CV_8UC3, imageData);
        Mat output;
        Mat transformMatrix = Mat(3, 3, CV_64F, perspectiveMatrix);

        cv::warpPerspective(input, output, transformMatrix, Size(outputWidth, outputHeight),
                            cv::INTER_CUBIC, cv::BORDER_REPLICATE);

        // Checking if the output image data is stored in a continuous block of memory
        if (output.isContinuous()) {
            // Total number of bytes in the output image
            size_t bytes = output.total() * output.elemSize();
            // Copying the output image data to the output buffer
            std::memcpy(outputBuffer, output.data, bytes);
        }
    }


    // Function to find contours in an image and store the point coordinates in a buffer
    __attribute__((visibility("default"))) __attribute__((used))
    int findContours(
            uchar* imageData,
            int imageWidth,
            int imageHeight,
            int* outputBuffer,
            int outputBufferSize
    ) {
        cv::Mat input(imageHeight, imageWidth, CV_8UC1, imageData);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(input, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

        int idx = 0;
        for (const auto& contour : contours) {
            for (const auto& point : contour) {
                // Ensuring enough space in buffer for point and end marker
                if (idx < outputBufferSize - 4) { // -4 to save space for the marker and next point
                    outputBuffer[idx++] = point.x;
                    outputBuffer[idx++] = point.y;
                }
            }
            // Marker to indicate the end of a contour
            if (idx < outputBufferSize - 4) {
                outputBuffer[idx++] = -1;  // -1 indicates end of a contour
                outputBuffer[idx++] = -1;
            }
        }
        return idx;
    }



    // Function to find the minimum area rectangle enclosing a set of points
    __attribute__((visibility("default"))) __attribute__((used))
    void minAreaRect(
            int* points,    // Array of points (x0, y0, x1, y1, ...)
            int numPoints,  // Number of points
            double* output  // Array for results (cx, cy, width, height, angle)
    ) {
        std::vector<cv::Point> contour;
        for (int i = 0; i < numPoints; i += 2) {
            // Adding points to the contour
            contour.push_back(cv::Point(points[i], points[i + 1]));
        }

        cv::RotatedRect rect = cv::minAreaRect(contour);

        // Storing the results in the output array
        output[0] = rect.center.x;
        output[1] = rect.center.y;
        output[2] = rect.size.width;
        output[3] = rect.size.height;
        output[4] = rect.angle;
    }



    // Function to fill a polygon in a mask
    __attribute__((visibility("default"))) __attribute__((used))
    void fillPoly(
            uchar* maskData,
            int maskWidth,
            int maskHeight,
            int* boxPoints,
            int numPoints // Nombre total de points dans boxPoints
    ) {
        cv::Mat mask(maskHeight, maskWidth, CV_8UC1, maskData);

        std::vector<std::vector<cv::Point>> polygon(1);
        for (int i = 0; i < numPoints; i += 2) {
            // Add points to the polygon
            polygon[0].push_back(cv::Point(boxPoints[i], boxPoints[i + 1]));
        }

        cv::fillPoly(mask, polygon, cv::Scalar(1));
    }


    // Function to preprocess the image for ocr
    __attribute__((visibility("default"))) __attribute__((used))
    void preProcessImage(
            uchar* imageData,
            int imageWidth,
            int imageHeight,
            uchar* outputData // Buffer to store the output image
    ) {
        cv::Mat input(imageHeight, imageWidth, CV_8UC3, imageData);
        cv::Mat labImage, processedImage, grayImage, filteredImage;

        // Convert to LAB Color model
        cv::cvtColor(input, labImage, cv::COLOR_BGR2Lab);

        // Splitting the LAB image to different channels
        std::vector<cv::Mat> labChannels(3);
        cv::split(labImage, labChannels);

        // Apply CLAHE to channel L
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(labChannels[0], labChannels[0]);

        // Merge the CLAHE enhanced L-channel with the a and b channel
        cv::merge(labChannels, labImage);

        // Convert to RGB Color model
        cv::cvtColor(labImage, processedImage, cv::COLOR_Lab2BGR);

        // Apply detailEnhance
        cv::detailEnhance(processedImage, processedImage, 40, 0.90);

        // Convert to grayscale
        cv::cvtColor(processedImage, grayImage, cv::COLOR_BGR2GRAY);

        // Apply bilateralFilter
        cv::bilateralFilter(grayImage, filteredImage, 45, 75, 75);

        std::memcpy(outputData, filteredImage.data, filteredImage.total() * filteredImage.elemSize());
    }

}