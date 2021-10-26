/**
 * @File    : pointSet_minOutsourcing.cpp
 * @Brief   : 点集的最小外包：圆形、直立矩阵、旋转矩阵、三角形、凸多边形
 * @Author  : Wei Li
 * @Date    : 2021-10-18
*/

#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


// -------------------------------
int main(int argc, char **argv)
{
    std::cout << "--------------------------------------------------" << std::endl;
    // -------- 点集的最小外包 --------
    // step 1, 最小外包旋转矩形
    // point set
    // way 1 of point set store.
    // cv::Mat pointSet = (cv::Mat_<float>(5, 2) << 1, 1, 5, 1, 1, 10, 5, 10, 2, 5);

    // way 2 of point set store.
    // std::vector<cv::Point2f> pointSet;
    // pointSet.push_back(cv::Point2f(1, 1));
    // pointSet.push_back(cv::Point2f(5, 1));
    // pointSet.push_back(cv::Point2f(1, 10));
    // pointSet.push_back(cv::Point2f(5, 10));
    // pointSet.push_back(cv::Point2f(2, 5));

    // way 3 of point set store.
    cv::Mat pointSet = (cv::Mat_<cv::Vec2f>(5, 1) << cv::Vec2f(1, 1), cv::Vec2f(5, 1), cv::Vec2f(1, 10), cv::Vec2f(5, 10), cv::Vec2f(2, 5));

    // compute the minimum outsourcing rotate rectangle of point set
    cv::RotatedRect rotateRectangle = cv::minAreaRect(pointSet);
    // show infomation of rotate rectangle
    std::cout << "Angel of rotate rectangle: " << rotateRectangle.angle << std::endl;
    std::cout << "Center of rotate rectangle: " << rotateRectangle.center << std::endl;
    std::cout << "Size of rotate rectangle: " << rotateRectangle.size << std::endl;

    // 旋转矩形是通过中心点坐标、尺寸和旋转角度三个方面来定义的，
    // 通过这三个属性值就可以计算出旋转矩形的 4 个顶点，这样虽然简单，但是写起来比较复杂。
    // rotate rectangel
    cv::RotatedRect rotateRect (cv::Point2f(200, 200), cv::Point2f(90, 150), -60);
    //计算旋转矩形的 4 个顶点，存储为一个 4 行 2 列的单通道 float 类型的 Mat
    cv::Mat vertices;
    cv::boxPoints(rotateRect, vertices);
    std::cout << vertices << std::endl;
    // 在黑色发布上绘制该旋转矩形
    cv::Mat img = cv::Mat::zeros(cv::Size(400, 400), CV_8UC1);
    for (size_t i = 0; i < 4; ++i)
    {
        cv::Point point_1 = static_cast<cv::Point>(vertices.row(i));
        int j = (i + 1) % 4;
        cv::Point point_2 = static_cast<cv::Point>(vertices.row(j));
        cv::line(img, point_1, point_2, cv::Scalar(255), 3);
    }
    cv::imshow("RotateRectangel", img);

    std::cout << "--------------------------------------------------" << std::endl;
    // step 2, 最小外包圆形
    cv::Mat pointSetCircle = (cv::Mat_<float>(5, 2) << 1, 1, 5, 1, 1, 10, 5, 10, 2, 5);
    // compute the minimum outsourcing circle with this point set
    cv::Point2f center_circle;
    float radius_circle;
    cv::minEnclosingCircle(pointSetCircle, center_circle, radius_circle);
    // show infomation of the minimum outsourcing circle
    std::cout << "The center of circle: " << center_circle << std::endl;
    std::cout << "The radius of circle: " << radius_circle << std::endl;

    std::cout << "--------------------------------------------------" << std::endl;
    // step 3, 最小外包直立矩形
    std::vector<cv::Point2f> pointSetRect;
    pointSetRect.push_back(cv::Point2f(1, 1));
    pointSetRect.push_back(cv::Point2f(5, 1));
    pointSetRect.push_back(cv::Point2f(1, 10));
    pointSetRect.push_back(cv::Point2f(5, 10));
    pointSetRect.push_back(cv::Point2f(2, 5));

    // compute theThe smallest outsourcing upright rectangle with this point set
    cv::Rect uprightRectangle = cv::boundingRect(cv::Mat(pointSetRect));
    // show infomation of this upright rectangle
    std::cout << "The infomation of upright rectangle: " << uprightRectangle <<  std::endl;

    std::cout << "--------------------------------------------------" << std::endl;
    // step 4, 最小凸包
    // 5 行 2 列的单通道 Mat
    cv::Mat pointSetConvex = (cv::Mat_<float>(5, 2) << 1, 1, 5, 1, 1, 10, 5, 10, 2, 5);
    // compute the convex outsourcing of this poin set
    std::vector<cv::Point2f> hull;
    cv::convexHull(pointSetConvex, hull);
    // 打印最外侧的点(连接起来即为凸多边形)
    for (size_t i = 0; i < hull.size(); ++i)
    {
        std::cout << hull[i] << ", ";
    }
    std::cout << std::endl;

    std::cout << "--------------------------------------------------" << std::endl;
    // step 5, 最小外包三角形
    cv::Mat pointSetTriangle = (cv::Mat_<float>(5, 2) << 1, 1, 5, 1, 1, 10, 5, 10, 2, 5);
    pointSetTriangle = pointSetTriangle.reshape(2, 5); // 转换为双通道矩阵
    std::vector<cv::Point> triangle; // 存储三角形三个顶点
    double areaTriangle = cv::minEnclosingTriangle(pointSetTriangle, triangle);
    std::cout << "The infomation of triangle: " << std::endl;
    for (size_t i = 0; i < 3; ++i)
    {
        std::cout << triangle[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "The area of triangle: " << areaTriangle << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    
    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
