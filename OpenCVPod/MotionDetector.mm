//
//  OpenCVWrapper.m
//  Camapp
//

#import "MotionDetector.h"
#import <AVFoundation/AVFoundation.h>
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

@interface MotionDetector() {
  
}
  
  @property (nonatomic, assign) BOOL isCapturing;
  
@end

@interface UIImage (UIImage_OpenCV)
  
+ (UIImage *)imageWithCVMat:(const cv::Mat&)cvMat;
+ (UIImage *)imageFromSampleBuffer:(CMSampleBufferRef)sampleBuffer;
- (cv::Mat)CVMat;
  
@end

@implementation MotionDetector
  
  - (instancetype)init {
    if (self = [super init]) {
      buf.resize(2);
    }
    return self;
  }
  
  // various tracking parameters (in seconds)
  const double MHI_DURATION = 5;
  const double MAX_TIME_DELTA = 0.5;
  const double MIN_TIME_DELTA = 0.15;
  
  // number of cyclic frame buffer used for motion detection
  
  // (should, probably, depend on FPS)
  
  const int N = 4;
  
  // ring image buffer
  
  vector<Mat> buf;
  
  int last = 0;
  
  // temporary images
  Mat mhi, orient, mask, segmask, zplane;
  vector<cv::Rect> regions;
  
  - (BOOL)checkImageForMotion:(const Mat&)img diffThreshold:(int)diffThreshold {
    double timestamp = (double)clock() / CLOCKS_PER_SEC; // get current time in seconds
    cv::Size size = img.size();
    int i, idx1 = last;
    cv::Rect comp_rect;
    double count;
    double angle;
    cv::Point center;
    double magnitude;
    Scalar color;
    Mat dst;
    
    // allocate images at the beginning or
    // reallocate them if the frame size is changed
    if (mhi.size() != size) {
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
    
    threshold(silh, silh, diffThreshold, 1, THRESH_BINARY); // and threshold it
    updateMotionHistory(silh, mhi, timestamp, MHI_DURATION); // update MHI
    
    // convert MHI to blue 8u image
    mhi.convertTo(mask, CV_8UC1, 255. / MHI_DURATION, (MHI_DURATION - timestamp)*255. / MHI_DURATION);
    
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
      } else { // i-th motion component
        comp_rect = regions[i];
        // reject very small components
        if (comp_rect.width + comp_rect.height < 100) {
          continue;
        }
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
      if (count < comp_rect.width*comp_rect.height * 0.05) {
        continue;
      }
      return YES;
    }
    return NO;
  }
  
  - (void)searchMotionWithBuffer:(CMSampleBufferRef)sampleBuffer motionBlock:(void (^)(BOOL))motionBlock {
    dispatch_async(dispatch_get_global_queue( DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(void){
      Mat image = [self sampleBufferToMat:sampleBuffer];
      BOOL motionDetected = [self checkImageForMotion:image diffThreshold:30];
      dispatch_async(dispatch_get_main_queue(), ^(void){
        if (motionBlock) {
          motionBlock(motionDetected);
        }
      });
    });
  }
  
  - (void)stopCapturing {
    self.isCapturing = NO;
  }
  
  - (Mat)sampleBufferToMat:(CMSampleBufferRef)sampleBuffer {
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    CVPixelBufferLockBaseAddress(imageBuffer, 0);
    
    void* bufferAddress;
    size_t width;
    size_t height;
    size_t bytesPerRow;
    
    int format_opencv;
    
    OSType format = CVPixelBufferGetPixelFormatType(imageBuffer);
    if (format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange) {
      format_opencv = CV_8UC1;

      bufferAddress = CVPixelBufferGetBaseAddressOfPlane(imageBuffer, 0);
      width = CVPixelBufferGetWidthOfPlane(imageBuffer, 0);
      height = CVPixelBufferGetHeightOfPlane(imageBuffer, 0);
      bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(imageBuffer, 0);
    } else { // expect kCVPixelFormatType_32BGRA
      format_opencv = CV_8UC4;

      bufferAddress = CVPixelBufferGetBaseAddress(imageBuffer);
      width = CVPixelBufferGetWidth(imageBuffer);
      height = CVPixelBufferGetHeight(imageBuffer);
      bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
    
    }
    Mat image = Mat((int)height, (int)width, format_opencv, bufferAddress, bytesPerRow);
    
    CVPixelBufferUnlockBaseAddress( imageBuffer, 0 );
    return image;
  }
  
@end

@implementation UIImage (UIImage_OpenCV)
  
  + (UIImage *)imageFromSampleBuffer:(CMSampleBufferRef)sampleBuffer {
    // Get a CMSampleBuffer's Core Video image buffer for the media data
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    
    // Lock the base address of the pixel buffer
    CVPixelBufferLockBaseAddress(imageBuffer, 0);
    
    // Get the number of bytes per row for the pixel buffer
    void *baseAddress = CVPixelBufferGetBaseAddress(imageBuffer);
    
    // Get the number of bytes per row for the pixel buffer
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
        
    // Get the pixel buffer width and height
    size_t width = CVPixelBufferGetWidth(imageBuffer);
    size_t height = CVPixelBufferGetHeight(imageBuffer);
    
    // Create a device-dependent RGB color space
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    
    // Create a bitmap graphics context with the sample buffer data
    CGContextRef context = CGBitmapContextCreate(baseAddress,
                                                 width,
                                                 height,
                                                 8,
                                                 bytesPerRow,
                                                 colorSpace,
                                                 kCGBitmapByteOrder32Little
                                                 | kCGImageAlphaPremultipliedFirst);
    // Create a Quartz image from the pixel data in the bitmap graphics context
    CGImageRef quartzImage = CGBitmapContextCreateImage(context);
    // Unlock the pixel buffer
    CVPixelBufferUnlockBaseAddress(imageBuffer,0);
    
    // Free up the context and color space
    CGContextRelease(context);
    CGColorSpaceRelease(colorSpace);
    
    // Create an image object from the Quartz image
    //UIImage *image = [UIImage imageWithCGImage:quartzImage];
    UIImage *image = [UIImage imageWithCGImage:quartzImage
                                         scale:1.0f
                                   orientation:UIImageOrientationRight];
    
    // Release the Quartz image
    CGImageRelease(quartzImage);
    
    return image;
  }
  
  + (UIImage *)imageFromIplImage:(IplImage *)image {
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
  
  - (id)initWithCVMat:(const cv::Mat&)cvMat {
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
