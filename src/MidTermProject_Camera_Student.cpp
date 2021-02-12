/* INCLUDES FOR THIS PROJECT */
#include <string>
#include <cstdlib>

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

// struct for evaluation
struct perfStats {
  std::string detectorType;
  std::string descriptorType;
  std::string matchingType;
  std::string selectorType;
  vector<int>   numKeyPointsPerframe;
  vector<int>   numKeyPointsPerROI;
  vector<int>   numMatchedKeyPoints;
  vector<double> detectorTime;
  vector<double> descriptorTime;
  vector<double> MatcherTime;
};

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */
	 std::vector<perfStats> combinations;
    perfStats curStat;
    vector<int>    numKeyPointsPerframe;
    vector<int>    numKeyPointsPerROI;
    vector<int>    numMatchedKeyPoints;
    vector<double> detectorTime;
    vector<double> descriptorTime;
    vector<double> MatcherTime;

    vector<std::string> detectorNames = {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
    vector<std::string> descriptorNames = {"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"};

	string detectorType;
    string descriptorType;
    string matcherType      = "MAT_BF";        // MAT_BF, MAT_FLANN
    string descriptorForm   = "DES_BINARY"; // DES_BINARY, DES_HOG
    string selectorType     = "SEL_KNN";       // SEL_NN, SEL_KNN

    for (int icomb = 0; icomb < detectorNames.size()*descriptorNames.size();++icomb) {  

        detectorType    = detectorNames[icomb/descriptorNames.size()];
        descriptorType  = descriptorNames[icomb%descriptorNames.size()];
        cout << endl;
        cout << "combination " << detectorType << " && " << descriptorType <<endl;
        cout << "=============================================" <<endl;
        if (detectorType.compare("AKAZE") != 0 && descriptorType.compare("AKAZE") == 0) {
            cout << "SKIPPING since AKAZE descriptor works only with KAZE or AKAZE keypoints"  <<endl;
            continue;
        } else if (detectorType.compare("SIFT") == 0 && descriptorType.compare("ORB") == 0) {
            cout << "SKIPPING since ORB descriptor does not work SIFT keypoints"  <<endl;
            continue;
        }
      
        // data location
        string dataPath = "../";

        // camera
        string imgBasePath = dataPath + "images/";
        string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
        string imgFileType = ".png";
        int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
        int imgEndIndex = 9;   // last file index to load
        int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

        // misc
        int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
        vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
        bool bVis = false;            // visualize results

        /* MAIN LOOP OVER ALL IMAGES */

        for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
        {
            /* LOAD IMAGE INTO BUFFER */

            // assemble filenames for current index
            ostringstream imgNumber;
            imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
            string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

            // load image from file and convert to grayscale
            cv::Mat img, imgGray;
            img = cv::imread(imgFullFilename);
            cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

            //// STUDENT ASSIGNMENT
            //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize
            // push image into data frame buffer
            DataFrame frame;
            frame.cameraImg = imgGray;
            dataBuffer.push_back(frame);
            // remove first element to keep size of ring buffer according to dataBufferSize
            if (dataBuffer.size() > dataBufferSize) {
                dataBuffer.erase(dataBuffer.begin());
            }

            //// EOF STUDENT ASSIGNMENT
            cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

            /* DETECT IMAGE KEYPOINTS */

            // extract 2D keypoints from current image
            vector<cv::KeyPoint> keypoints; // create empty feature list for current image
            //// STUDENT ASSIGNMENT
            //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
            //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

            detectorTime.push_back((double)cv::getTickCount());
            if (detectorType.compare("SHITOMASI") == 0)
            {
                detKeypointsShiTomasi(keypoints, imgGray, false);
            }
            else if (detectorType.compare("ORB") == 0) 
            {
                cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
                detector->detect(imgGray,keypoints);
                cout << "loaded ORB detector" << endl;
            }
            else if (detectorType.compare("FAST") == 0) 
            {
                cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
                detector->detect(imgGray,keypoints);
                cout << "loaded FAST detector" << endl;
            }
                else if (detectorType.compare("AKAZE") == 0) 
            {
                cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();
                detector->detect(imgGray,keypoints);
                cout << "loaded AKAZE detector" << endl;
            }
            else if (detectorType.compare("SIFT") == 0) 
            {
                cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> detector = cv::xfeatures2d::SiftFeatureDetector::create();
                detector->detect(imgGray,keypoints);
                cout << "loaded SIFT detector" << endl;
            }        
            else if (detectorType.compare("BRISK") == 0) 
            {
                cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
                detector->detect(imgGray,keypoints);
            }
            else if (detectorType.compare("HARRIS") == 0)
            {
                // Detector parameters
                int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
                int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
                int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
                double k = 0.04;       // Harris parameter (see equation for details)

                // Detect Harris corners and normalize output
                cv::Mat dst, dst_norm, dst_norm_scaled;
                dst = cv::Mat::zeros(imgGray.size(), CV_32FC1);
                cv::cornerHarris(imgGray, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
                cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
                cv::convertScaleAbs(dst_norm, dst_norm_scaled);

                // Look for prominent corners and instantiate keypoints
                //vector<cv::KeyPoint> keypoints;
                double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
                for (size_t j = 0; j < dst_norm.rows; j++)
                {
                    for (size_t i = 0; i < dst_norm.cols; i++)
                    {
                        int response = (int)dst_norm.at<float>(j, i);
                        if (response > minResponse)
                        { // only store points above a threshold

                            cv::KeyPoint newKeyPoint;
                            newKeyPoint.pt = cv::Point2f(i, j);
                            newKeyPoint.size = 2 * apertureSize;
                            newKeyPoint.response = response;

                            // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                            bool bOverlap = false;
                            for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                            {
                                double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                                if (kptOverlap > maxOverlap)
                                {
                                    bOverlap = true;
                                    if (newKeyPoint.response > (*it).response)
                                    {                      // if overlap is >t AND response is higher for new kpt
                                        *it = newKeyPoint; // replace old key point with new one
                                        break;             // quit loop over keypoints
                                    }
                                }
                            }
                            if (!bOverlap)
                            {                                     // only add new key point if no overlap has been found in previous NMS
                                keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                            }
                        }
                    } // eof loop over cols
                }     // eof loop over rows  
            } 
            else
            {
                //...
                cout << "no detector found" << endl;
            }      
            detectorTime.back() = ((double)cv::getTickCount() - detectorTime.back()) / cv::getTickFrequency()*1000/1.0;

            //// EOF STUDENT ASSIGNMENT

            //// STUDENT ASSIGNMENT
            //// TASK MP.3 -> only keep keypoints on the preceding vehicle

            // only keep keypoints on the preceding vehicle
            numKeyPointsPerframe.push_back(keypoints.size());
            bool bFocusOnVehicle = true;
            cv::Rect vehicleRect(535, 180, 180, 150);
            if (bFocusOnVehicle)
            {
                // Iterate through keypoints 
                int i = 0;
                //cout << "Deleting keypoints outside of ROI | num keypoints:  " << keypoints.size() << endl;
                while (i < keypoints.size()) {
                    // check whether keypoint is i nboundaries 
                    if ((keypoints[i].pt.x < vehicleRect.x) || 
                        (keypoints[i].pt.x > vehicleRect.x + vehicleRect.width) ||
                        (keypoints[i].pt.y < vehicleRect.y) || 
                        (keypoints[i].pt.y > vehicleRect.y + vehicleRect.height) 
                        ) {
                        // delete keypoint
                        keypoints.erase(keypoints.begin()+i);
                    } else {
                        ++i;
                    }
                }
                //cout << "num remaining keypoints:  " << keypoints.size() << endl;
            }
            numKeyPointsPerROI.push_back(keypoints.size());

            //// EOF STUDENT ASSIGNMENT

            // optional : limit number of keypoints (helpful for debugging and learning)
            bool bLimitKpts = false;
            if (bLimitKpts)
            {
                int maxKeypoints = 50;

                if (detectorType.compare("SHITOMASI") == 0)
                { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                    keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                }
                cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                cout << " NOTE: Keypoints have been limited!" << endl;
            }

            // push keypoints and descriptor for current frame to end of data buffer
            (dataBuffer.end() - 1)->keypoints = keypoints;
            cout << "#2 : DETECT KEYPOINTS done" << endl;

            /* EXTRACT KEYPOINT DESCRIPTORS */

            //// STUDENT ASSIGNMENT
            //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
            //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

            cv::Mat descriptors;
            
            // Find descriptors
            descriptorTime.push_back((double)cv::getTickCount());
            descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
            descriptorTime.back() = ((double)cv::getTickCount() - descriptorTime.back()) / cv::getTickFrequency()*1000/1.0;
            //// EOF STUDENT ASSIGNMENT

            // push descriptors for current frame to end of data buffer
            (dataBuffer.end() - 1)->descriptors = descriptors;

            cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

            if (dataBuffer.size() > 1) // wait until at least two images have been processed
            {

                /* MATCH KEYPOINT DESCRIPTORS */

                vector<cv::DMatch> matches;
                matcherType     = "MAT_BF";         // MAT_BF, MAT_FLANN
              	if (descriptorType.compare("SIFT") == 0) {
                  descriptorForm  = "DES_HOG";     // DES_BINARY, DES_HOG
                }
                else {
                  descriptorForm  = "DES_BINARY";     // DES_BINARY, DES_HOG
                }
                    

                //// STUDENT ASSIGNMENT
                //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
                //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

                MatcherTime.push_back((double)cv::getTickCount());
                matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                matches, descriptorForm, matcherType, selectorType);
                MatcherTime.back() = ((double)cv::getTickCount() - MatcherTime.back()) / cv::getTickFrequency()*1000/1.0;
                numMatchedKeyPoints.push_back(matches.size());
                //// EOF STUDENT ASSIGNMENT

                // store matches in current data frame
                (dataBuffer.end() - 1)->kptMatches = matches;

                cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

                // visualize matches between current and previous image
                bVis = true;
                if (bVis)
                {
                    cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                    cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                    (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                    matches, matchImg,
                                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                                    vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                    string windowName = "Matching keypoints between two camera images";
                    cv::namedWindow(windowName, 7);
                    cv::imshow(windowName, matchImg);
                    // Save image of matching
                    if (imgIndex == 9) {
                        cv::Point2f top_left(50,50);
                        cv::Scalar font_color(0,0,255);
                      	double max = *max_element(numKeyPointsPerROI.begin(), numKeyPointsPerROI.end());
                      	double min = *min_element(numKeyPointsPerROI.begin(), numKeyPointsPerROI.end());
                        std::string overlay_text = "Det | Desc: " + detectorType + " | " + descriptorType
                            + " Num KPs in ROI: " + std::to_string((int) min)
                            + " to " + std::to_string((int) max);
                        cv::putText(matchImg, overlay_text, top_left, cv::FONT_HERSHEY_COMPLEX, 1, font_color);
                        cv::imwrite("../imgs/" + detectorType + "_" + descriptorType + "img" + std::to_string(imgIndex) + ".png",matchImg);
                    }

                    cout << "Press key to continue to next image" << endl;
//                    cv::waitKey(0); // wait for key to be pressed
                }
                bVis = false;
            } else {
                MatcherTime.push_back(0);
                numMatchedKeyPoints.push_back(0);
            }

        } // eof loop over all images
        // Fill in stats
        curStat.detectorType            = detectorType;
        curStat.descriptorType          = descriptorType;
        curStat.matchingType            = matcherType;
        curStat.selectorType            = selectorType;
        curStat.numKeyPointsPerframe    = numKeyPointsPerframe;
        curStat.numKeyPointsPerROI      = numKeyPointsPerROI;
        curStat.numMatchedKeyPoints     = numMatchedKeyPoints;
        curStat.detectorTime            = detectorTime;
        curStat.descriptorTime          = descriptorTime;
        curStat.MatcherTime             = MatcherTime;
        combinations.push_back(curStat);
      
		numKeyPointsPerframe.clear();
		numKeyPointsPerROI.clear();
		numMatchedKeyPoints.clear();
		detectorTime.clear();
		descriptorTime.clear();
		MatcherTime.clear();
      
    } // end of combining detectors/descriptors

  	//Evaluation: Save to CSV File
  
  std::string filename = "../data.csv";
  std::ofstream output_stream(filename, std::ios::binary);

  if (!output_stream.is_open()) {
    std::cerr << "failed to open file: " << filename << std::endl;
    return EXIT_FAILURE;
  }
  
  // write CSV header row
  output_stream << "Detector Type" << ","
                << "Descriptor Type" << ","
                << "Matching Type" << ","
                << "Frame#" << ","
                << "#KeyPointsPerFrame" << ","
                << "#KeyPointsPerROI" << ","
                << "DetectorTime(ms)" << ","
                << "DescriptorTime(ms)" << ","
                << "#MatchedPoints" << "," << "MatchingTime(ms))" << std::endl;
  
  // write det/des performance data to .csv output file line by line
  for (const auto &combo : combinations) {
    for (int i = 0; i < 10; i++) {
      output_stream << combo.detectorType
                    << "," << combo.descriptorType
                    << "," << combo.matchingType
                    << "," << i
                    << "," << combo.numKeyPointsPerframe[i]
                    << "," << combo.numKeyPointsPerROI[i]
                    << "," << std::fixed << std::setprecision(8) << combo.detectorTime[i]
                    << "," << std::fixed << std::setprecision(8) << combo.descriptorTime[i]
                    << "," << combo.numMatchedKeyPoints[i]
                    << "," << std::fixed << std::setprecision(8) << combo.MatcherTime[i] << std::endl;
    }
    output_stream << std::endl;
  }
  
  output_stream.close();
  
    return 0;
}
