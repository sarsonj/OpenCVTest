//
//  OpenCVWrapper.h
//  Camapp
//

#import <UIKit/UIKit.h>
#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>

@interface MotionDetector : NSObject
  
  - (void)stopCapturing;
  - (void)searchMotionWithBuffer:(CMSampleBufferRef)sampleBuffer motionBlock:(void (^)(BOOL))motionBlock;
@end
