using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using Size = OpenCvSharp.Size;

namespace camera_measure
{
    internal static class caliberation
    {
        static int BoardSize_Width = 9; //宽度和高度是指内部交叉点的个数，而不是方形格的个数
        static int BoardSize_Height = 6;
        static Size BoardSize = new Size(BoardSize_Width, BoardSize_Height);
        static int SquareSize = 10; //每格的宽度应设置为实际的毫米数
        static int winSize = 5;
        static Mat cameraMatrix = new Mat(), distCoeffs = new Mat();
        public static void Run(string[] files)
        {
            var img = Cv2.ImRead(files[0]);
            List<string> imagesList = new List<string>();
            foreach(var f in files)
            {
                imagesList.Add(f);
            }
            List<Point2f[]> imagesPoints = new List<Point2f[]>();
            
            Size imageSize = new Size();
            bool found = false;

            Mat[] imagesPointsM = new Mat[imagesList.Count];

            for(int i = 0; i < imagesList.Count; i++)
            {
                Mat view = Cv2.ImRead(imagesList[i]);
                if(!view.Empty())
                {
                    imageSize = view.Size();
                    Point2f[] pointBuf;

                    found = Cv2.FindChessboardCorners(view, BoardSize, out pointBuf, ChessboardFlags.AdaptiveThresh | ChessboardFlags.NormalizeImage);
                    if(found == true)
                    {
                        //灰度化
                        Mat viewGray = new Mat();
                        Cv2.CvtColor(view, viewGray, ColorConversionCodes.BGR2GRAY);
                        //亚像素精细化
                        Cv2.CornerSubPix(viewGray, pointBuf, new Size(winSize, winSize), new Size(-1, -1), new TermCriteria(CriteriaTypes.Eps | CriteriaTypes.Count, 30, 0.0001));
                        imagesPoints.Add(pointBuf);
                        Mat p = Mat.FromArray<Point2f>(pointBuf);
                        imagesPointsM[i] = p;
                        //画出角点
                        Cv2.DrawChessboardCorners(view, BoardSize, pointBuf, found);
                        Mat temp = view.Clone();
                        Cv2.ImShow("Image View", view);
                        Cv2.WaitKey(500);
                    }
                }
            }
            Mat[] rvecs = new Mat[0];//平移矩阵
            Mat[] tvecs = new Mat[0];//旋转矩阵
            //开始标定
            runCalibration(imagesList.Count, imageSize, out cameraMatrix, out distCoeffs, imagesPointsM, out rvecs, out tvecs, out double totalAvgErr);

            Console.WriteLine("Camera Matrix:\n{0}", Cv2.Format(cameraMatrix));
            Console.WriteLine("Distortion Coefficients:\n{0}", Cv2.Format(distCoeffs));
            Console.WriteLine("Total Average Error:\n{0}", totalAvgErr);
            //映射坐标
            Mat map1 = new Mat();
            Mat map2 = new Mat();
            Mat newCameraMatrix = Cv2.GetOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, out Rect roi);
            Cv2.InitUndistortRectifyMap(cameraMatrix, distCoeffs, new Mat(), newCameraMatrix, imageSize, MatType.CV_16SC2, map1, map2);
            var px = map1.At<byte>(0,0);
            for(int i = 0; i < imagesList.Count; i++)
            {
                Mat view = Cv2.ImRead(imagesList[i], ImreadModes.Color);
                Mat rview = new Mat();
                if(view.Empty())
                    continue;
                Cv2.Remap(view, rview, map1, map2, InterpolationFlags.Linear);
                Cv2.ImShow("Image View", rview);
                Cv2.WaitKey(500);
            }
            Cv2.WaitKey();
        }
        public static bool runCalibration(int imagesCount, Size imageSize, out Mat cameraMatrix, out Mat distCoeffs, Mat[] imagePoints, out Mat[] rvecs, out Mat[] tvecs, out double totalAvgErr)
        {
            cameraMatrix = Mat.Eye(new Size(3, 3), MatType.CV_64F);
            distCoeffs = Mat.Zeros(new Size(8, 1), MatType.CV_64F);
            //提取角点坐标
            Mat[] objectPoints = calcBoardCornerPositions(BoardSize, SquareSize, imagesCount);
            //Find intrinsic and extrinsic camera parameters
            double rms = Cv2.CalibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, out rvecs, out tvecs, CalibrationFlags.None);
            bool ok = Cv2.CheckRange(InputArray.Create(cameraMatrix)) && Cv2.CheckRange(InputArray.Create(distCoeffs));
            totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints, rvecs, tvecs, cameraMatrix, distCoeffs);
            return ok;
        }

        public static Mat[] calcBoardCornerPositions(Size BoardSize, float SquareSize, int imagesCount)
        {
            Mat[] corners = new Mat[imagesCount];
            for(int k = 0; k < imagesCount; k++)
            {
                Point3f[] p = new Point3f[BoardSize.Height * BoardSize.Width];

                for(int i = 0; i < BoardSize.Height; i++)
                {
                    for(int j = 0; j < BoardSize.Width; j++)
                    {
                        p[i * BoardSize.Width + j] = new Point3f(j * SquareSize, i * SquareSize, 0);
                    }
                }
                //corners[k] = new Mat(BoardSize.Width, BoardSize.Height, MatType.CV_64F, p); 
                corners[k] = Mat.FromArray<Point3f>(p);
            }
            return corners;
        }
        public static double computeReprojectionErrors(Mat[] objectPoints, Mat[] imagePoints, Mat[] rvecs, Mat[] tvecs, Mat cameraMatrix, Mat distCoeffs)
        {
            Mat imagePoints2 = new Mat();
            int totalPoints = 0;
            double totalErr = 0, err;

            for(int i = 0; i < objectPoints.Length; ++i)
            {
                Cv2.ProjectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);

                err = Cv2.Norm(imagePoints[i], imagePoints2, NormTypes.L2);

                int n = objectPoints[i].Width * objectPoints[i].Height;
                //perViewErrors[i] = (float)Math.Sqrt(err * err / n);
                totalErr += err * err;
                totalPoints += n;
            }

            return Math.Sqrt(totalErr / totalPoints);
        }
    }
}

