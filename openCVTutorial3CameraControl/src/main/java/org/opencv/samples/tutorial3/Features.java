package org.opencv.samples.tutorial3;

/**
 * Created by Jcama on 02/01/2016.
 */
public class Features {

    double area;
    int contourPoints;
    int numCircles;

    public Features(){

    }

    double getArea(){return this.area;}
    void setArea(double area){this.area = area;}

    double getContourPoints(){return this.contourPoints;}
    void setcontourPoints(int contourPoints){this.contourPoints = contourPoints;}

    int getNumCircles(){return this.numCircles;}
    void setNumCircles(int numCircles){this.numCircles = numCircles;}
}
