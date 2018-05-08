//
//  OpenCVWrapper.h
//  Camapp
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

@interface OpenCVWrapper : NSObject
  
  - (UIImage *)captureImage;
  + (NSString *)openCVVersionString;
  - (void)stopCapturing;
  - (void)startDetecting:(void (^)(UIImage *))motionBlock;

@end
