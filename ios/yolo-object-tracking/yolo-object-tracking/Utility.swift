//
//  utility.swift
//  yolo-object-tracking
//
//  Created by Mikael Von Holst on 2018-01-09.
//  Copyright © 2018 Mikael Von Holst. All rights reserved.
//

import Foundation
import Vision
import VideoToolbox

extension Double {
    func format(f: String) -> String {
        return String(format: "%\(f)f", self)
    }
}

func scaleCenterCropImage(pixelbuffer:CVPixelBuffer, imageSide:Int, scaleFactor:Double) -> CVPixelBuffer? {
    var convertedImage: CGImage?
    VTCreateCGImageFromCVPixelBuffer(pixelbuffer, nil, &convertedImage)
    if let cgImage = convertedImage {
        let context = CGContext(data: nil, width: imageSide, height: imageSide, bitsPerComponent: cgImage.bitsPerComponent, bytesPerRow: 0, space: cgImage.colorSpace!, bitmapInfo: cgImage.bitmapInfo.rawValue)
        
        // crop image to fit square image
        let drawHeight = Double(cgImage.height) * scaleFactor
        let x = (Double(cgImage.width) - drawHeight) / 2
        let y = (Double(cgImage.height) - drawHeight) / 2
        let rect = CGRect(x: x, y: y, width: drawHeight, height: drawHeight)
        
        // rescale image by drawing in context
        if let croppedImage = cgImage.cropping(to: rect) {
            context?.draw(croppedImage, in: CGRect(origin: CGPoint.zero, size: CGSize(width: imageSide, height: imageSide)))
            if let scaledImage = context!.makeImage() {
                
                // convert back to CVPixelbuffer
                let returnBuffer = pixelBufferFromCGImage(image: scaledImage)
                return returnBuffer
            }
        }
    }
    return nil
}

func pixelBufferFromCGImage(image: CGImage) -> CVPixelBuffer? {
    let ciImage = CIImage.init(cgImage: image)
    let ciContext = CIContext()
    var resizeBuffer: CVPixelBuffer?
    CVPixelBufferCreate(kCFAllocatorDefault, image.width, image.height, kCVPixelFormatType_32ARGB, nil, &resizeBuffer)
    ciContext.render(ciImage, to: resizeBuffer!)
    return resizeBuffer
}

