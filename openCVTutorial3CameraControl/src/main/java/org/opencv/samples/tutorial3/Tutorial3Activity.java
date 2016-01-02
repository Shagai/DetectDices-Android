package org.opencv.samples.tutorial3;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.ListIterator;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.hardware.Camera.Size;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.SubMenu;
import android.view.SurfaceView;
import android.view.View;
import android.view.View.OnTouchListener;
import android.view.WindowManager;
import android.widget.Toast;

public class Tutorial3Activity extends Activity implements CvCameraViewListener2, OnTouchListener {
    private static final String TAG = "OCVSample::Activity";

    private Tutorial3View mOpenCvCameraView;
    private List<Size> mResolutionList;
    private MenuItem[] mEffectMenuItems;
    private SubMenu mColorEffectsMenu;
    private MenuItem[] mResolutionMenuItems;
    private SubMenu mResolutionMenu;

    private int                 index = 0;
    private DetectDices         det = new DetectDices();
    private boolean task = false;
    private List<MatOfPoint> squares = null;
    private Mat circles = null;
    private String[] letter;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    mOpenCvCameraView.setOnTouchListener(Tutorial3Activity.this);
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public Tutorial3Activity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.tutorial3_surface_view);

        mOpenCvCameraView = (Tutorial3View) findViewById(R.id.tutorial3_activity_java_surface_view);

        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);

        mOpenCvCameraView.setCvCameraViewListener(this);

    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
    }

    public void onCameraViewStopped() {
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        Mat rgba = inputFrame.rgba();
        final Mat gray = inputFrame.gray();
        org.opencv.core.Size sizeRgba = rgba.size();

        Mat rgbaInnerWindow;

        int rows = (int) sizeRgba.height;
        int cols = (int) sizeRgba.width;

        if (this.task == false) {
            det.execute(rgba, gray);
            this.task = true;
        }

        List<MatOfPoint> squares = det.GetSquares();
        Mat cirlces = det.GetCircles();
        String[] letterss = det.GetLetter();
        if (!det.isBusy()) {
            //rgba = frame.clone(); // Desremove //  when you want fluid video
            det = new DetectDices();
            this.task = false;
            this.squares = squares;     // Remove when you want fluid video
            this.circles = cirlces;
            this.letter = letterss;
        }

        Imgproc.drawContours(rgba, this.squares, -1, new Scalar(0, 255, 0), 3);
        if (this.circles != null) {
            for (int x = 0; x < this.circles.cols(); x++) {
                double vCircle[] = circles.get(0, x);

                if (vCircle == null)
                    break;

                Point pt = new Point(Math.round(vCircle[0]), Math.round(vCircle[1]));
                int radius = (int) Math.round(vCircle[2]);

                // draw the found circle
                Imgproc.circle(rgba, pt, radius, new Scalar(0, 255, 0), 3);
                Imgproc.circle(rgba, pt, 3, new Scalar(0, 0, 255), 3);
                for (int i = 0; i < this.squares.size(); i++) {
                    Imgproc.putText(rgba, this.letter[i], new Point(this.squares.get(i).get(2, 0)),
                            Core.FONT_HERSHEY_SIMPLEX, 3, new Scalar(255, 255, 255), 2);
                }
            }
        }
        //Imgproc.drawContours(gray, this.squares, -1, new Scalar(255, 255, 255), -1);
        //DetectRed(rgba);

        //CleanBackground(rgba, this.squares, rgba);
        return rgba;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {

        return true;
    }

    public boolean onOptionsItemSelected(MenuItem item) {

        return true;
    }

    @SuppressLint("SimpleDateFormat")
    @Override
    public boolean onTouch(View v, MotionEvent event) {
        mOpenCvCameraView.setupCameraFlashLight();
        return false;
    }

    private void DetectRed(Mat rgba){
        if (rgba.cols() != 0) {
            Mat imgHSV = new Mat();
            //Mat erorde_ker = new Mat().ones(1, 1, CvType.CV_32F);
            Imgproc.cvtColor(rgba, imgHSV, Imgproc.COLOR_BGR2HSV);
            //Core.inRange(imgHSV, new Scalar(0, 100, 100), new Scalar(20, 255, 255), rgba);
            Core.inRange(imgHSV, new Scalar(115, 125, 125), new Scalar(130, 255, 255), rgba);
            if (imgHSV.cols() != 0) {
                List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
                Mat hierarchy = new Mat();
                Mat chan = rgba.clone();
                Imgproc.findContours(chan, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
                double area = 0;
                for (int i = 0; i < contours.size(); i++) {
                    area += Imgproc.contourArea(contours.get(contours.size() - 1));
                }
                //System.out.println("AREAAAA: " + area);
            }
            //Imgproc.erode(rgba, rgba, erorde_ker);
            //Imgproc.dilate(rgba, rgba, erorde_ker);
        }
    }

    private void CleanBackground(Mat img, List<MatOfPoint> squares, Mat res){
        Mat mask = new Mat().zeros(new org.opencv.core.Size(img.cols(), img.rows()), CvType.CV_8UC1);
        Imgproc.drawContours(mask, squares, -1, new Scalar(255, 255, 255), -1);
        img.copyTo(res, mask);
    }
}
