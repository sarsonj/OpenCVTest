//
//  OpenCVWrapper.m
//  Camapp
//

#import "OpenCVWrapper.h"
#include "opencv2/optflow.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <time.h>
#include <stdio.h>
#include <ctype.h>

using namespace cv;
using namespace std;
using namespace cv::motempl;

@interface UIImage (UIImage_OpenCV)
  
+ (UIImage *)imageWithCVMat:(const cv::Mat&)cvMat;
  
@end

@interface OpenCVWrapper()
{
  CascadeClassifier cascade;
}
  
@property (nonatomic, assign) BOOL isCapturing;
  
  
@end

@implementation OpenCVWrapper

  - (instancetype)init {
    if (self = [super init]) {
      buf.resize(2);
    }
    return self;
  }
  
  + (NSString *)openCVVersionString {
    return [NSString stringWithFormat:@"OpenCV Version %s",  CV_VERSION];
  }

  // various tracking parameters (in seconds)
  const double MHI_DURATION = 5;
  const double MAX_TIME_DELTA = 0.5;
  const double MIN_TIME_DELTA = 0.05;
  
  // number of cyclic frame buffer used for motion detection
  
  // (should, probably, depend on FPS)
  
  const int N = 4;
  
  // ring image buffer
  
  vector<Mat> buf;
  
  
  int last = 0;
  
  // temporary images
  
  CvMemStorage* storage = 0; // temporary storage
  
  // parameters:
  
  //  img – input video frame
  
  //  dst – resultant motion picture
  
  //  args – optional parameters
  
  /*
  void update_mhi(IplImage* img, IplImage* dst, int diff_threshold) {
    double timestamp = (double)clock()/CLOCKS_PER_SEC; // get current time in seconds
    CvSize size = cvSize(img->width,img->height); // get current frame size
    int i, idx1 = last, idx2;
    IplImage* silh;
    CvSeq* seq;
    CvRect comp_rect;
    double count;
    double angle;
    CvPoint center;
    double magnitude;
    CvScalar color;
    
    // allocate images at the beginning or
    
    // reallocate them if the frame size is changed
    
    if(!mhi || mhi->width != size.width || mhi->height != size.height) {
      if( buf == 0 ) {
        buf = (IplImage**)malloc(N*sizeof(buf[0]));
        memset(buf, 0, N*sizeof(buf[0]));
      }
      for(i = 0; i < N; i++) {
        cvReleaseImage(&buf[i]);
        buf[i] = cvCreateImage( size, IPL_DEPTH_8U, 1);
        cvZero( buf[i] );
      }
      cvReleaseImage(&mhi);
      cvReleaseImage(&orient);
      cvReleaseImage(&segmask);
      cvReleaseImage(&mask);
      
      mhi = cvCreateImage( size, IPL_DEPTH_32F, 1 );
      
      cvZero( mhi ); // clear MHI at the beginning
      
      orient = cvCreateImage( size, IPL_DEPTH_32F, 1 );
      
      segmask = cvCreateImage( size, IPL_DEPTH_32F, 1 );
      
      mask = cvCreateImage( size, IPL_DEPTH_8U, 1 );
      
    }
    
    cvCvtColor( img, buf[last], CV_BGR2GRAY ); // convert frame to grayscale
    
    idx2 = (last + 1) % N; // index of (last – (N-1))th frame
    
    last = idx2;
    
    silh = buf[idx2];
    
    cvAbsDiff( buf[idx1], buf[idx2], silh ); // get difference between frames
    
    cvThreshold( silh, silh, diff_threshold, 1, CV_THRESH_BINARY ); // and threshold it

    
    cv::motempl::updateMotionHistory(silh, mhi, timestamp, MHI_DURATION); // update MHI
//    updateMotionHistory( silh, mhi, timestamp, MHI_DURATION ); // update MHI
    

  
    // convert MHI to blue 8u image
    
    cvCvtScale(mhi, mask, 255./MHI_DURATION, (MHI_DURATION - timestamp)*255./MHI_DURATION);
    
    cvZero(dst);
    
    cvMerge(mask, 0, 0, 0, dst);
    
    
    
    // calculate motion gradient orientation and valid orientation mask
    
    cvCalcMotionGradient( mhi, mask, orient, MAX_TIME_DELTA, MIN_TIME_DELTA, 3 );
    
    
    
    if( !storage )
    
    storage = cvCreateMemStorage(0);
    
    else
    
    cvClearMemStorage(storage);
    
    
    
    // segment motion: get sequence of motion components
    
    // segmask is marked motion components map. It is not used further
    
    seq = cvSegmentMotion( mhi, segmask, storage, timestamp, MAX_TIME_DELTA );
    
    
    
    // iterate through the motion components,
    
    // One more iteration (i == -1) corresponds to the whole image (global motion)
    
    for( i = -1; i < seq->total; i++ ) {
      
      
      
      if( i < 0 ) { // case of the whole image
        
        comp_rect = cvRect( 0, 0, size.width, size.height );
        
        color = CV_RGB(255,255,255);
        
        magnitude = 100;
        
      }
      
      else { // i-th motion component
        
        comp_rect = ((CvConnectedComp*)cvGetSeqElem( seq, i ))->rect;
        
        if( comp_rect.width + comp_rect.height < 100 ) // reject very small components
        
        continue;
        
        color = CV_RGB(255,0,0);
        
        magnitude = 30;
        
      }
      
      
      
      // select component ROI
      
      cvSetImageROI( silh, comp_rect );
      
      cvSetImageROI( mhi, comp_rect );
      
      cvSetImageROI( orient, comp_rect );
      
      cvSetImageROI( mask, comp_rect );
      
      
      
      // calculate orientation
      
//      angle = cvCalcGlobalOrientation( orient, mask, mhi, timestamp, MHI_DURATION);
      
      angle = 360.0 - angle;  // adjust for images with top-left origin
      
      
      
      count = cvNorm( silh, 0, CV_L1, 0 ); // calculate number of points within silhouette ROI
      
      
      
      cvResetImageROI( mhi );
      
      cvResetImageROI( orient );
      
      cvResetImageROI( mask );
      
      cvResetImageROI( silh );
      
      
      
      // check for the case of little motion
      
      if( count < comp_rect.width*comp_rect.height * 0.05 )
      
      continue;
      
      
      
      // draw a clock with arrow indicating the direction
      
      center = cvPoint( (comp_rect.x + comp_rect.width/2),
                       
                       (comp_rect.y + comp_rect.height/2) );
      
      
      
      cvCircle( dst, center, cvRound(magnitude*1.2), color, 3, CV_AA, 0 );
      
      cvLine( dst, center, cvPoint( cvRound( center.x + magnitude*cos(angle*CV_PI/180)),
                                   
                                   cvRound( center.y - magnitude*sin(angle*CV_PI/180))), color, 3, CV_AA, 0 );
      
    }
  }
   */
  
  Mat mhi, orient, mask, segmask, zplane;
  vector<cv::Rect> regions;
  
  void update_mhi(const Mat& img, Mat& dst, int diff_threshold)
  {
    double timestamp = (double)clock() / CLOCKS_PER_SEC; // get current time in seconds
    cv::Size size = img.size();
    int i, idx1 = last;
    cv::Rect comp_rect;
    double count;
    double angle;
    cv::Point center;
    double magnitude;
    Scalar color;
    
    // allocate images at the beginning or
    // reallocate them if the frame size is changed
    if (mhi.size() != size)
    {
      mhi = Mat::zeros(size, CV_32F);
      zplane = Mat::zeros(size, CV_8U);
      
      buf[0] = Mat::zeros(size, CV_8U);
      buf[1] = Mat::zeros(size, CV_8U);
    }
    
    cvtColor(img, buf[last], COLOR_BGR2GRAY); // convert frame to grayscale
    
    int idx2 = (last + 1) % 2; // index of (last - (N-1))th frame
    last = idx2;
    
    Mat silh = buf[idx2];
    absdiff(buf[idx1], buf[idx2], silh); // get difference between frames
    
    threshold(silh, silh, diff_threshold, 1, THRESH_BINARY); // and threshold it
    updateMotionHistory(silh, mhi, timestamp, MHI_DURATION); // update MHI
    
    // convert MHI to blue 8u image
    mhi.convertTo(mask, CV_8U, 255. / MHI_DURATION, (MHI_DURATION - timestamp)*255. / MHI_DURATION);
    
    Mat planes[] = { mask, zplane, zplane };
    merge(planes, 3, dst);
    
    // calculate motion gradient orientation and valid orientation mask
    calcMotionGradient(mhi, mask, orient, MAX_TIME_DELTA, MIN_TIME_DELTA, 3);
    
    // segment motion: get sequence of motion components
    // segmask is marked motion components map. It is not used further
    regions.clear();
    segmentMotion(mhi, segmask, regions, timestamp, MAX_TIME_DELTA);
    
    // iterate through the motion components,
    // One more iteration (i == -1) corresponds to the whole image (global motion)
    for (i = -1; i < (int)regions.size(); i++) {
      
      if (i < 0) { // case of the whole image
        comp_rect = cv::Rect(0, 0, size.width, size.height);
        color = Scalar(255, 255, 255);
        magnitude = 100;
      }
      else { // i-th motion component
        comp_rect = regions[i];
        if (comp_rect.width + comp_rect.height < 100) // reject very small components
        continue;
        color = Scalar(0, 0, 255);
        magnitude = 30;
      }
      
      // select component ROI
      Mat silh_roi = silh(comp_rect);
      Mat mhi_roi = mhi(comp_rect);
      Mat orient_roi = orient(comp_rect);
      Mat mask_roi = mask(comp_rect);
      
      // calculate orientation
      angle = calcGlobalOrientation(orient_roi, mask_roi, mhi_roi, timestamp, MHI_DURATION);
      angle = 360.0 - angle;  // adjust for images with top-left origin
      
      count = norm(silh_roi, NORM_L1); // calculate number of points within silhouette ROI
      
      // check for the case of little motion
      if (count < comp_rect.width*comp_rect.height * 0.05)
      continue;
      
      // draw a clock with arrow indicating the direction
      center = cv::Point((comp_rect.x + comp_rect.width / 2),
                     (comp_rect.y + comp_rect.height / 2));
      
      circle(img, center, cvRound(magnitude*1.2), color, 3, 16, 0);
      line(img, center, cv::Point(cvRound(center.x + magnitude*cos(angle*CV_PI / 180)),
                              cvRound(center.y - magnitude*sin(angle*CV_PI / 180))), color, 3, 16, 0);
    }
  }
  
  - (void)startDetecting:(void (^)(UIImage *))motionBlock {
    [self checkMotion:motionBlock];
  }
  
  - (void)checkMotion:(void (^)(UIImage *))motionBlock {
    CvCapture *capture = cvCreateCameraCapture(0);
    dispatch_async(dispatch_get_global_queue( DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(void){
      Mat image, motion;
      IplImage *captureImage = cvQueryFrame(capture);
      image = cvarrToMat(captureImage);
      if (image.empty()) {
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 1 * NSEC_PER_SEC), dispatch_get_main_queue(), ^{
          [self checkMotion:motionBlock];
        });
      }
      update_mhi(image, motion, 30);
      UIImage *motionImage = [UIImage imageWithCVMat:image];
      dispatch_async(dispatch_get_main_queue(), ^(void){
        if (!motion.empty()) {
          if (motionBlock) {
            motionBlock(motionImage);
          }
          printf("asdasdasdasd");
        }
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 0.25 * NSEC_PER_SEC), dispatch_get_main_queue(), ^{
          [self checkMotion:motionBlock];
        });
      });
    });
  }
  
  - (void)stopCapturing {
    self.isCapturing = NO;
  }
  
  - (UIImage *)captureImage {
    CvCapture *capture = cvCreateCameraCapture(0);
    IplImage *image = cvQueryFrame(capture);
    return [self UIImageFromIplImage:image];
  }
    // NOTE you SHOULD cvReleaseImage() for the return value when end of the code.
  - (IplImage *)CreateIplImageFromUIImage:(UIImage *)image {
    // Getting CGImage from UIImage
    CGImageRef imageRef = image.CGImage;
    
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    // Creating temporal IplImage for drawing
    IplImage *iplimage = cvCreateImage(
                                       cvSize(image.size.width,image.size.height), IPL_DEPTH_8U, 4
                                       );
    // Creating CGContext for temporal IplImage
    CGContextRef contextRef = CGBitmapContextCreate(
                                                    iplimage->imageData, iplimage->width, iplimage->height,
                                                    iplimage->depth, iplimage->widthStep,
                                                    colorSpace, kCGImageAlphaPremultipliedLast|kCGBitmapByteOrderDefault
                                                    );
    // Drawing CGImage to CGContext
    CGContextDrawImage(
                       contextRef,
                       CGRectMake(0, 0, image.size.width, image.size.height),
                       imageRef
                       );
    CGContextRelease(contextRef);
    CGColorSpaceRelease(colorSpace);
    
    // Creating result IplImage
    IplImage *ret = cvCreateImage(cvGetSize(iplimage), IPL_DEPTH_8U, 3);
    cvCvtColor(iplimage, ret, CV_RGBA2BGR);
    cvReleaseImage(&iplimage);
    
    return ret;
  }
  
    // NOTE You should convert color mode as RGB before passing to this function
  - (UIImage *)UIImageFromIplImage:(IplImage *)image {
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    // Allocating the buffer for CGImage
    NSData *data =
    [NSData dataWithBytes:image->imageData length:image->imageSize];
    CGDataProviderRef provider =
    CGDataProviderCreateWithCFData((CFDataRef)data);
    // Creating CGImage from chunk of IplImage
    CGImageRef imageRef = CGImageCreate(
                                        image->width, image->height,
                                        image->depth, image->depth * image->nChannels, image->widthStep,
                                        colorSpace, kCGImageAlphaNone|kCGBitmapByteOrderDefault,
                                        provider, NULL, false, kCGRenderingIntentDefault
                                        );
    // Getting UIImage from CGImage
    UIImage *ret = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    return ret;
  }
@end

@implementation UIImage (UIImage_OpenCV)
  
  - (cv::Mat)CVMat {
    
      CGColorSpaceRef colorSpace = CGImageGetColorSpace(self.CGImage);
      CGFloat cols = self.size.width;
      CGFloat rows = self.size.height;
    
      cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels
    
      CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to backing data
                                                      cols,                      // Width of bitmap
                                                      rows,                     // Height of bitmap
                                                      8,                          // Bits per component
                                                      cvMat.step[0],              // Bytes per row
                                                      colorSpace,                 // Colorspace
                                                      kCGImageAlphaNoneSkipLast |
                                                      kCGBitmapByteOrderDefault); // Bitmap info flags
    
      CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), self.CGImage);
      CGContextRelease(contextRef);
    
      return cvMat;
  }
  
  - (cv::Mat)CVGrayscaleMat {
      CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceGray();
      CGFloat cols = self.size.width;
      CGFloat rows = self.size.height;
    
      cv::Mat cvMat = cv::Mat(rows, cols, CV_8UC1); // 8 bits per component, 1 channel
    
      CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to backing data
                                                      cols,                      // Width of bitmap
                                                      rows,                     // Height of bitmap
                                                      8,                          // Bits per component
                                                      cvMat.step[0],              // Bytes per row
                                                      colorSpace,                 // Colorspace
                                                      kCGImageAlphaNone |
                                                      kCGBitmapByteOrderDefault); // Bitmap info flags
    
      CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), self.CGImage);
      CGContextRelease(contextRef);
      CGColorSpaceRelease(colorSpace);
    
      return cvMat;
  }
  
  + (UIImage *)imageWithCVMat:(const cv::Mat&)cvMat {
      return [[UIImage alloc] initWithCVMat:cvMat];
    }
  
  - (instancetype)initWithCVMat:(const cv::Mat&)cvMat {
      NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize() * cvMat.total()];
    
      CGColorSpaceRef colorSpace;
    
      if (cvMat.elemSize() == 1)
      {
        colorSpace = CGColorSpaceCreateDeviceGray();
      }
      else
      {
        colorSpace = CGColorSpaceCreateDeviceRGB();
      }
    
      CGDataProviderRef provider = CGDataProviderCreateWithCFData((CFDataRef)data);
    
      CGImageRef imageRef = CGImageCreate(cvMat.cols,                                     // Width
                                          cvMat.rows,                                     // Height
                                          8,                                              // Bits per component
                                          8 * cvMat.elemSize(),                           // Bits per pixel
                                          cvMat.step[0],                                  // Bytes per row
                                          colorSpace,                                     // Colorspace
                                          kCGImageAlphaNone | kCGBitmapByteOrderDefault,  // Bitmap info flags
                                          provider,                                       // CGDataProviderRef
                                          NULL,                                           // Decode
                                          false,                                          // Should interpolate
                                          kCGRenderingIntentDefault);                     // Intent
    
      self = [self initWithCGImage:imageRef];
      CGImageRelease(imageRef);
      CGDataProviderRelease(provider);
      CGColorSpaceRelease(colorSpace);
    
      return self;
  }
  
@end
