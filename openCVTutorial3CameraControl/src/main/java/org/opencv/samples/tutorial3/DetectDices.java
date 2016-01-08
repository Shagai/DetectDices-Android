package org.opencv.samples.tutorial3;

import android.os.AsyncTask;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by Jcama on 29/12/2015.
 */

public class DetectDices extends AsyncTask<Mat, Void, Void> {

    private boolean busy = true;
    private Mat frame = null;
    private List<MatOfPoint> squares = null;
    private Mat circles = null;
    private String[] letter = {"-1", "-1", "-1", "-1", "-1"};

    public DetectDices() {

    }

    public Void doInBackground(Mat... params) {
        mainDetect(params[0], params[1]);
        this.busy = false;
        return null;
    }

    public Mat GetFrame(){
        return this.frame;
    }

    public List<MatOfPoint> GetSquares(){
        return this.squares;
    }

    public boolean isBusy() {return this.busy;}

    public void mainDetect(Mat rgba, Mat gray){
        Size sizeRgba = rgba.size();
        Mat rgbaInnerWindow;
        int rows = (int) sizeRgba.height;
        int cols = (int) sizeRgba.width;
        Mat cl1 = new Mat();
        rgbaInnerWindow = rgba.submat(0, rows, 0, cols);
        Equalizer(gray, cl1);
        Blur(cl1, rgbaInnerWindow);
        Dilate_Erode(rgbaInnerWindow, cl1);
        this.squares = SquaresFilter(Find_Squares(cl1));
        DetectFeatures(this.squares, gray, rgba);
    }

    private void Equalizer(Mat img, Mat cl1) {
        Mat mattemp = new Mat();
        double clipLimit = 1.2;
        Size tileGridSize = new Size(8, 8);
        CLAHE clahe = Imgproc.createCLAHE(clipLimit, tileGridSize);
        clahe.apply(img, mattemp);
        Imgproc.equalizeHist(mattemp, cl1);
    }

    private void Blur(Mat img, Mat blurDst) {
        Size kernel = new Size(5, 5);
        Imgproc.blur(img, blurDst, kernel);
    }

    private void Dilate_Erode(Mat img, Mat erodeDst) {
        Mat dilated_ker = Mat.ones(1, 1, CvType.CV_32F);
        Mat erorde_ker = Mat.ones(6, 6, CvType.CV_32F);
        Mat dilateDst = new Mat();
        Imgproc.dilate(img, dilateDst, dilated_ker);
        Imgproc.erode(dilateDst, erodeDst, erorde_ker);
    }

    private List<MatOfPoint> Find_Squares(Mat img) {
        Mat gaussDst = new Mat();
        Size gaussKer = new Size(5, 5);
        Imgproc.GaussianBlur(img, gaussDst, gaussKer, 0);
        List<MatOfPoint> contours = new ArrayList<>();
        List<MatOfPoint> squares = new ArrayList<>();

        Mat bin = new Mat();
        List<Mat> listGrays = new ArrayList<>();
        Core.split(gaussDst, listGrays);
        for (Mat gray : listGrays) {
            for (int thrs = 0; thrs <= 255; thrs += 26) {
                if (thrs == 0) {
                    Mat cannyDst = new Mat();
                    //int apertureSize = 5;
                    Imgproc.Canny(img, cannyDst, 0, 50);
                    Mat dilated_ker = Mat.ones(1, 1, CvType.CV_32F);
                    Imgproc.dilate(cannyDst, bin, dilated_ker);
                } else {
                    Imgproc.threshold(gray, bin, thrs, 255, Imgproc.THRESH_BINARY);
                }

                Mat hierarchy = new Mat();
                Imgproc.findContours(bin, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
                for (MatOfPoint cnt : contours) {
                    MatOfPoint2f cnt2f = new MatOfPoint2f(cnt.toArray());
                    double cnt_len = Imgproc.arcLength(cnt2f, true);
                    Imgproc.approxPolyDP(cnt2f, cnt2f, 0.02 * cnt_len, true);
                    cnt2f.convertTo(cnt, CvType.CV_32S);
                    if ((cnt.height() == 4) && Imgproc.contourArea(cnt2f) > 1000 &&
                            Imgproc.contourArea(cnt2f) < 100000 && Imgproc.isContourConvex(cnt)) {

                        double[] cos = new double[4];

                        for (int i = 0; i < 4; i++){
                            double[] p0 = cnt.get(i, 0);
                            double[] p1 = cnt.get((i + 1)%4, 0);
                            double[] p2 = cnt.get((i + 2)%4, 0);
                            cos[i] = angle_cos(p0, p1, p2);
                        }

                        if (getMax(cos) < 0.1){
                            squares.add(cnt);
                        }
                    }
                }
            }
        }

        return squares;
    }

    private double angle_cos(double[] p0, double[] p1, double[] p2){
        double[] d1 = new double[2];
        double[] d2 = new double[2];

        for (int i = 0; i < 2; i++) {
            d1[i] = p0[i] - p1[i];
            d2[i] = p2[i] - p1[i];
        }

        double dot = d1[0] * d2[0] + d1[1] * d2[1];
        double dot1 = d1[0] * d1[0] + d1[1] * d1[1];
        double dot2 = d2[0] * d2[0] + d2[1] * d2[1];

        return Math.abs(dot / Math.sqrt(dot1 * dot2));
    }

    private static double getMax(double[] array){
        double max = 0;
        for (double num : array) {
            if (num > max) {
                max = num;
            }
        }

        return max;
    }

    private List<MatOfPoint> SquaresFilter(List<MatOfPoint> squares) {
        List<MatOfPoint> filters = new ArrayList<>();
        boolean[] sq = new boolean[squares.size()];
        Arrays.fill(sq, true);
        for (int i = 0; i < squares.size(); i++){
            if (sq[i]){
                filters.add(squares.get(i));
                sq[i] = false;
            }
            for (int j = 0; j < squares.size(); j++){
                if (sq[j]){
                    double[] centerI = Center(squares.get(i)).get(0, 0);
                    double[] centerJ = Center(squares.get(j)).get(0,0);
                    if (Math.abs(centerI[0] - centerJ[0]) < 20 &&
                            Math.abs(centerI[1] - centerJ[1]) < 20){
                        sq[j] = false;
                        break;
                    }
                }
            }
        }

        return filters;
    }

    private MatOfPoint Center(MatOfPoint square){
        Point mid = new Point();
        mid.x = (square.get(0,0)[0] + square.get(2,0)[0]) / 2;
        mid.y = (square.get(0,0)[1] + square.get(2,0)[1]) / 2;
        return new MatOfPoint(mid);
    }

    private void DetectFeatures(List<MatOfPoint> squares, Mat gray, Mat rgba){
        for (int i = 0; i  < squares.size(); i++){
            //TODO: Detectar J y Q. Probar detectando negro, areas de contorno y círculos
            Mat color = new Mat();
            Mat matDest = new Mat();
            Mat circles = new Mat();
            CleanBackground(gray, rgba, squares.get(i), matDest, color);
            Features features = new Features();
            DetectRed(color, features);
            Detect_Circles(matDest, squares.get(i), features);
            if (features.getArea() > 0 && features.getNumCircles() >= 5){
                this.letter[i] = "8";
                continue;
            }
            if (features.getArea() == 0 && features.getNumCircles() >= 5){
                this.letter[i] = "7";
                continue;
            }
            if (features.getArea() > 250 && features.getNumCircles() < 3 &&
                    features.getContourPoints() < 100){
                this.letter[i] = "As";
                continue;
            }
            if (features.getArea() < 300 && features.getContourPoints() > 85){
                this.letter[i] = "K";
                continue;
            }
            if (features.getArea() > 10){
                this.letter[i] = "K";
                continue;
            }
            DetectBlack(matDest, features);
            if (features.getArea() > 1000){
                this.letter[i] = "Q";
                continue;
            }
            if (features.getArea() < 500){
                this.letter[i] = "J";
                continue;
            }
            if (features.getNumCircles() > 1){
                this.letter[i] = "Q";
                continue;
            }
            else{
                this.letter[i] = "J";
            }

        }
    }

    private void CleanBackground(Mat img, Mat img2, MatOfPoint square, Mat res, Mat rgba){
        List<MatOfPoint> squares = new ArrayList<>();
        squares.add(square);
        Mat mask = Mat.zeros(new Size(img.cols(), img.rows()), CvType.CV_8UC1);
        Imgproc.drawContours(mask, squares, -1, new Scalar(255, 255, 255), -1);
        img.copyTo(res, mask);
        img2.copyTo(rgba, mask);
    }

    private void Detect_Circles(Mat img, MatOfPoint square, Features features){
        double dp = 0.5;
        double minDist = 15;
        double param1 = 80.0;
        double param2 = 10.0;
        int minRadius = 1;
        int maxRadius = 10;
        Mat circles = new Mat();
        Imgproc.HoughCircles(img, circles, Imgproc.CV_HOUGH_GRADIENT, dp, minDist, param1, param2,
                minRadius, maxRadius);
        features.setNumCircles(circles.cols());
    }

    public Mat GetCircles()
    {
        return this.circles;
    }

    private double DetectRed(Mat rgba, Features features){
        double area = -1;
        int contourPoints = -1;
        if (rgba.cols() != 0) {
            Mat imgHSV = new Mat();
            Mat chan = new Mat();
            Mat res = new Mat();
            //Mat erorde_ker = new Mat().ones(1, 1, CvType.CV_32F);
            Imgproc.cvtColor(rgba, imgHSV, Imgproc.COLOR_BGR2HSV);
            //Core.inRange(imgHSV, new Scalar(0, 100, 100), new Scalar(20, 255, 255), rgba);
            Core.inRange(imgHSV, new Scalar(115, 125, 125), new Scalar(140, 255, 255), chan);
            if (imgHSV.cols() != 0) {
                List<MatOfPoint> contours = new ArrayList<>();
                Mat hierarchy = new Mat();
                Imgproc.threshold(chan, res, 127, 255, Imgproc.THRESH_BINARY);
                Imgproc.findContours(res, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
                area = 0;
                contourPoints = 0;
                for (MatOfPoint contour : contours) {
                    double ar = Imgproc.contourArea(contour);
                    if (ar > 0) {
                        area += ar;
                        contourPoints += contour.rows();
                    }
                }
                if (area == 0 && contours.size() > 5){
                    area = 100;
                }
                //System.out.println("AREAAAA: " + area);
            }
            features.setArea(area);
            features.setcontourPoints(contourPoints);
            //Imgproc.erode(rgba, rgba, erorde_ker);
            //Imgproc.dilate(rgba, rgba, erorde_ker);
        }
        return area;
    }

    private double DetectBlack(Mat img, Features features){
        Mat res = new Mat();
        double area = -1;
        int contourPoints = -1;
        if (img.cols() != 0) {
            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.threshold(img, res, 127, 255, Imgproc.THRESH_BINARY);
            Imgproc.findContours(res, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
            area = 0;
            contourPoints = 0;
            for (MatOfPoint contour : contours) {
                double ar = Imgproc.contourArea(contour);
                if (ar < 3000 && ar > 0) {
                    area += ar;
                    contourPoints += contour.rows();
                }
            }
            if (area == 0 && contours.size() > 5){
                area = 100;
            }
            features.setArea(area);
            features.setcontourPoints(contourPoints);
        }
        return area;
    }

    public String[] GetLetter(){ return this.letter;}
}

