

#include <iostream>
#include <stdio.h>
#include <time.h>

#include <opencv2\opencv.hpp>



/*  OpenCVの名前空間を使用する  */
using namespace cv;
using namespace std;

// 共通変数
static int NumOfImg; // z方向の画像枚数
static int ImgX;
static int ImgY;
static char file_name[256];

static clock_t start;
static clock_t time_end;

Mat Org_Img;//元画像
Mat Hsv_Img;//HSV系に変更
Mat Rgb_Img;//RGB系
Mat Mask_Img;//mask後の画像
Mat Mask_Img2;//mask後の画像2
Mat Wise_Img;//論理積
Mat Gray_Image;//グレースケール
Mat Label_Img;//ラベル付け用
Mat Bin_Img;//二値画像(フィルタ後)
Mat Bin_Img2;//二値画像
Mat Result_Img;//ラベル付け画像
Mat stats;
Mat centroids;

cv::Point2f center,center2, p1;


//最小座標を求める
cv::Point minPoint(vector<cv::Point> contours) {
	double minx = contours.at(0).x;
	double miny = contours.at(0).y;
	for (int i = 1; i < contours.size(); i++) {
		if (minx > contours.at(i).x) {
			minx = contours.at(i).x;
		}
		if (miny > contours.at(i).y) {
			miny = contours.at(i).y;
		}
	}
	return cv::Point(minx, miny);
}
//最大座標を求める
cv::Point maxPoint(vector<cv::Point> contours) {
	double maxx = contours.at(0).x;
	double maxy = contours.at(0).y;
	for (int i = 1; i < contours.size(); i++) {
		if (maxx < contours.at(i).x) {
			maxx = contours.at(i).x;
		}
		if (maxy < contours.at(i).y) {
			maxy = contours.at(i).y;
		}
	}
	return cv::Point(maxx, maxy);
}



int main(int argc, char* argv[])
{
	start = clock();//実行時間計測開始
	int loop = 100;
	

  for (int i = 0; i < loop; i++) {
	float radius;
	//画像赤
	//Org_Img = imread("./input/2019-10-27-211204.jpg", IMREAD_UNCHANGED); 
	//画像青
	Org_Img = imread("./input/2019-10-27-211134.jpg", IMREAD_UNCHANGED);

	// 画像が読み込まれなかったらプログラム終了
	if (Org_Img.empty()) return -1;

	//HSV
	//cv::cvtColor(Org_Img,Hsv_Img,COLOR_BGR2HSV);
	//RGB
	cv::cvtColor(Org_Img, Rgb_Img, COLOR_BGR2RGB);
	//RGB(GRAY)
	//cv::cvtColor(Org_Img, Rgb_Img, COLOR_BGR2GRAY);

	//マスク処理（赤）
	// cv::inRange(Hsv_Img, cv::Scalar(150, 100, 180, 0), cv::Scalar(180, 255, 255, 0), Mask_Img);
	// cv::inRange(Hsv_Img, cv::Scalar(0, 0, 180, 0), cv::Scalar(30, 255, 255, 0), Mask_Img2);

	//cv::inRange(Rgb_Img, cv::Scalar(255, 0, 0, 0), cv::Scalar(255, 255, 255, 0), Mask_Img);

	//マスク処理（青）
	//cv::inRange(Hsv_Img, cv::Scalar(100, 200, 220, 0), cv::Scalar(120, 255, 255, 0), Mask_Img);
	cv::inRange(Rgb_Img, cv::Scalar(0, 0, 253, 0), cv::Scalar(255, 255, 255, 0), Mask_Img);

	//GRAYの2値化
	//threshold(Rgb_Img, Mask_Img, 33, 255, THRESH_BINARY);


	// Bin_Img2 = Mask_Img + Mask_Img2;//HSV
	Bin_Img2 = Mask_Img;//RGB

	//中央値フィルタ
	cv::GaussianBlur(Bin_Img2, Bin_Img ,Size(3,3),0);

	// 輪郭を格納するcontoursにfindContours関数に渡すと輪郭を点の集合として入れてくれる
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(Bin_Img, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);    // 輪郭線を格納


	try {
		// 各輪郭をcontourArea関数に渡し、最大面積を持つ輪郭を探す
		double max_area = 0;
		double max_area2 = 0;
		int max_area_contour = -1;
		int max_area_contour2 = -1;
		for (int j = 0; j < contours.size(); j++) {
			double area = cv::contourArea(contours.at(j));		
			if (max_area < area) {
				max_area = area;
				max_area_contour = j;
			}
			else if (max_area2 < area) {
				max_area2 = area;
				max_area_contour2 = j;
			}
		}

		

		// 最大面積を持つ輪郭の最小外接円を取得
		cv::minEnclosingCircle(contours.at(max_area_contour), center, radius);
		cv::minEnclosingCircle(contours.at(max_area_contour2), center2, radius);

		cv::Point maxP = maxPoint(contours.at(max_area_contour));
		cv::Point minP = minPoint(contours.at(max_area_contour));
		cv::Point maxP2 = maxPoint(contours.at(max_area_contour2));
		cv::Point minP2 = minPoint(contours.at(max_area_contour2));
		maxP.x = center.x;
		minP.x = center.x;
		maxP2.x = center2.x;
		minP2.x = center2.x;
		cv::line(Org_Img, maxP, minP, cv::Scalar(0, 0, 255), 3, 4);
		cv::line(Org_Img, maxP2, minP2, cv::Scalar(0, 0, 255), 3, 4);
		cv::line(Org_Img, maxP, maxP2, cv::Scalar(0, 0, 255), 3, 4);
		cv::line(Org_Img, minP, minP2, cv::Scalar(0, 0, 255), 3, 4);

		// 最小外接円を描画
	    //cv::circle(Org_Img, center, radius, cv::Scalar(0, 0, 255), 3, 4);
		//cv::circle(, center, radius, cv::Scalar(0, 0, 255), 3, 4);
	    //cv::circle(Bin_Img, center, radius, cv::Scalar(0, 0, 255), 3, 4);
	}
	catch (ErrorCallback) {

	}

  }
	 time_end = clock();//実行時間計測終了

	 const double time = static_cast<double>(time_end - start) / CLOCKS_PER_SEC * 1000 / loop;
	 printf("実行時間：%f(fps)\n", 1000/time);


	//結果表示
	cv::namedWindow("Result",WINDOW_AUTOSIZE|WINDOW_FREERATIO);
	cv::imshow("Result", Org_Img);
	cv::imshow("Result2", Bin_Img);
	cv::waitKey(0);
}

