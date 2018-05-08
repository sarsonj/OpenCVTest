//
//  ViewController.swift
//  OpenCVPod
//
//  Created by iOS Developer on 03/05/2018.
//  Copyright © 2018 Space-O. All rights reserved.
//

import UIKit

class ViewController: UIViewController {

  @IBOutlet private var imageView: UIImageView!
  let wrapper = OpenCVWrapper()
  
  override func viewDidLoad() {
    super.viewDidLoad()
    self.imageView.image = wrapper.captureImage()
    wrapper.startDetecting { [unowned self] image in
      self.imageView.image = image
    }
  }

}

