//
//  ViewController.swift
//  yolo-object-tracking
//
//  Created by Mikael Von Holst on 2017-12-19.
//  Copyright © 2017 Mikael Von Holst. All rights reserved.
//

import UIKit
import CoreML
import Vision
import AVFoundation
import Accelerate

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    @IBOutlet weak var cameraView: UIView!
    @IBOutlet weak var frameLabel: UILabel!
    private var trackingRequest : VNRequest?
    let semaphore = DispatchSemaphore(value: 1)
    var lastExecution = Date()
    var screenHeight: Double?
    var screenWidth: Double?

    
    private lazy var cameraLayer: AVCaptureVideoPreviewLayer = AVCaptureVideoPreviewLayer(session: self.captureSession)
    private lazy var captureSession: AVCaptureSession = {
        let session = AVCaptureSession()
        session.sessionPreset = AVCaptureSession.Preset.hd1280x720
        guard let backCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            return session
        }

        for format in backCamera.formats {
            let ranges = format.videoSupportedFrameRateRanges as [AVFrameRateRange]
            var frameRates = ranges[0]
            
            // 3
            if frameRates.maxFrameRate == 240 {
                
                // 4
                do {
                    try backCamera.lockForConfiguration()
                    backCamera.activeFormat = format as AVCaptureDevice.Format
                    backCamera.activeVideoMinFrameDuration = frameRates.minFrameDuration
                    backCamera.activeVideoMaxFrameDuration = frameRates.maxFrameDuration
                    backCamera.unlockForConfiguration()
                    
                } catch {
                    
                }
            }
        }

        guard let input = try? AVCaptureDeviceInput(device: backCamera)
            else { return session }
        
        session.addInput(input)
        return session
    }()

    let numBoxes = 7*7*5 // mobilenet features
    let classLabels = ["white_dude", "black_dude"]
    let anchors:[Double] = [ 0.3, 0.3, 1.975, 1.975, 3.65, 3.65, 5.325, 5.325, 7.0, 7.0 ]

    let selectHowMany = 3
    let selectPerClass = 1
    
    let scoreThreshold: Float = 0.5
    let iouThreshold: Float = 0.3
    var confidenceInClassThreshold:Double  {
        return Double(0.8 * scoreThreshold)
    }

    var boundingBoxes: [BoundingBox] = []
    let multiClass = true
    
    override func viewDidLoad() {
        super.viewDidLoad()
        self.cameraView?.layer.addSublayer(self.cameraLayer)
        self.cameraView?.bringSubview(toFront: self.frameLabel)
        self.frameLabel.textAlignment = .left
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "MyQueue"))
        self.captureSession.addOutput(videoOutput)
        self.captureSession.startRunning()
        setupVision()

        setupBoxes()
        
        screenWidth = Double(view.frame.width)
        screenHeight = Double(view.frame.height)
        
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        cameraLayer.frame = cameraView.layer.bounds
    }
    
    func setupBoxes() {
        // Create shape layers for the bounding boxes.
        for _ in 0..<numBoxes {
            let box = BoundingBox()
            box.addToLayer(view.layer)
            self.boundingBoxes.append(box)
        }
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        self.cameraLayer.frame = self.cameraView?.bounds ?? .zero
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

    func setupVision() {
        guard let visionModel = try? VNCoreMLModel(for: traq_object_detector().model)
            else { fatalError("Can't load VisionML model") }
        let trackingRequest = VNCoreMLRequest(model: visionModel, completionHandler: processClassifications)
        trackingRequest.imageCropAndScaleOption = VNImageCropAndScaleOption.scaleFill
        self.trackingRequest = trackingRequest
    }
    
    func processClassifications(for request: VNRequest, error: Error?) {
        self.semaphore.signal()
        let thisExecution = Date()
        let executionTime = thisExecution.timeIntervalSince(lastExecution)
        let framesPerSecond:Double = 1/executionTime
        lastExecution = thisExecution
        DispatchQueue.main.async {
            guard let results = request.results as? [VNCoreMLFeatureValueObservation] else {
                return
            }
            guard let first = results.first else {
                return
            }
            guard let array = first.featureValue.multiArrayValue else {
                return
            }
            self.frameLabel.text = "FPS: \(framesPerSecond.format(f: ".3"))"

            let m = MultiArray<Double>(array)
            let predictions = self.predictionsFromMultiArray(array: m)
            let filteredIndices = predictions.indices.filter { predictions[$0].score > self.scoreThreshold }
            var selected:[Int]
            if self.multiClass {
                selected = nonMaxSuppressionMultiClass(numClasses: self.classLabels.count,
                                                           predictions: predictions,
                                                           scoreThreshold: self.scoreThreshold,
                                                           iouThreshold: self.iouThreshold,
                                                           maxPerClass: self.selectPerClass,
                                                           maxTotal: self.selectHowMany)
            } else {
                selected = nonMaxSuppression(predictions: predictions,
                                             indices: filteredIndices,
                                             iouThreshold: self.iouThreshold,
                                             maxBoxes: self.selectHowMany)
            }
            
            for i in 0..<self.numBoxes {
                var color: UIColor
                let textColor: UIColor
                if selected.contains(i) {
                    let prediction = predictions[i]
                    let textLabel = String(format: "%.2f - %@", prediction.score, self.classLabels[prediction.classIndex])
                    switch prediction.classIndex {
                    case 0:
                        color = UIColor.red
                    case 1:
                        color = UIColor.blue
                    case 2:
                        color = UIColor.purple
                    case 3:
                        color = UIColor.cyan
                    case 4:
                        color = UIColor.gray
                    case 5:
                        color = UIColor.green
                    case 6:
                        color = UIColor.yellow
                    case 7:
                        color = UIColor.white
                    case 8:
                        color = UIColor.magenta
                    default:
                        print("other class")
                        color = UIColor(red: 1.0, green: 0.0, blue: 0.0, alpha: 1)
                    }
                    color = color.withAlphaComponent(CGFloat(prediction.score))
                    if prediction.classIndex == 0 {
                        
                    }
                    
                    textColor = UIColor.black
                    self.boundingBoxes[i].show(frame: prediction.rect,
                                               label: textLabel,
                                               color: color, textColor: textColor)
                } else {
                    self.boundingBoxes[i].hide()
                }
            }

        }
    }
    
    func predictionsFromMultiArray(array:MultiArray<Double>) -> [NMSPrediction] {
        var predictions = [NMSPrediction]()
        let valuesPerCell = classLabels.count + 5
        let cellBoxes = Int(array.shape[0] / valuesPerCell)
        let cellRows = array.shape[1]
        let cellCols = array.shape[2]
        let numClasses = classLabels.count

        for r in 0..<cellRows {
            for c in 0..<cellCols {
                for b in 0..<cellBoxes {
                    
                    let startIndex = b * valuesPerCell
                    let tx = array[startIndex + 0, r, c]
                    let ty = array[startIndex + 1, r, c]
                    let tw = array[startIndex + 2, r, c]
                    let th = array[startIndex + 3, r, c]
                    let tc = array[startIndex + 4, r, c]
                    var classes = [Double](repeating: 0, count: numClasses)
                    for cl in 0..<numClasses {
                        classes[cl] = array[startIndex + 5 + cl, r, c]
                    }
                    let confidence = sigmoid(tc)
                    let confidenceInClassList = softmax(classes)
                    let (detectedClass, bestClassScore) = confidenceInClassList.argmax()
                    let confidenceInClass = confidence * bestClassScore
                    if confidenceInClass > confidenceInClassThreshold {
                        let boxW = anchors[2 * b + 0] * exp(tw) / Double(cellCols) * screenWidth!
                        let boxH = anchors[2 * b + 1] * exp(th) / Double(cellRows) * screenHeight!
                        let boxXCenter = ((Double(c) + sigmoid(tx)) / Double(cellCols)) * screenWidth!
                        let boxYCenter = ((Double(r) + sigmoid(ty)) / Double(cellRows)) * screenHeight!
                        
                        //works in portrait, needs separate model for landscape
                        let boxX = boxXCenter - boxW/2
                        let boxY = boxYCenter - boxH/2 //+ (screenHeight! - screenWidth!)/2
                        let rect = CGRect(x: boxX, y: boxY, width: boxW, height: boxH)
                        predictions.append((detectedClass, Float(confidence), rect))
                    }
                }
            }
        }
        return predictions
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        guard let trackingRequest = self.trackingRequest else { return }
        
        var requestOptions:[VNImageOption : Any] = [:]
        if let cameraIntrinsicData = CMGetAttachment(sampleBuffer, kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, nil) {
            requestOptions = [.cameraIntrinsics:cameraIntrinsicData]
        }
        self.semaphore.wait()

        DispatchQueue.global(qos: .background).async {
            
            //let exifOrientation = self.compensatingEXIFOrientation(deviceOrientation: UIDevice.current.orientation)
            let orientation = CGImagePropertyOrientation(rawValue: UInt32(EXIFOrientation.rightTop.rawValue))

            let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation!, options: requestOptions)
            do {
                try imageRequestHandler.perform([trackingRequest])
            } catch {
                print(error)
            }
        }
    }

    func sigmoid(_ val:Double) -> Double {
         return 1.0/(1.0 + exp(-val))
    }

    func softmax(_ values:[Double]) -> [Double] {
        if values.count == 1 { return [1.0]}
        guard let maxValue = values.max() else {
            fatalError("Softmax error")
        }
        let expValues = values.map { exp($0 - maxValue)}
        let expSum = expValues.reduce(0, +)
        return expValues.map({$0/expSum})
    }
    
    public static func softmax2(_ x: [Double]) -> [Double] {
        var x:[Float] = x.flatMap{Float($0)}
        let len = vDSP_Length(x.count)
        
        // Find the maximum value in the input array.
        var max: Float = 0
        vDSP_maxv(x, 1, &max, len)
        
        // Subtract the maximum from all the elements in the array.
        // Now the highest value in the array is 0.
        max = -max
        vDSP_vsadd(x, 1, &max, &x, 1, len)
        
        // Exponentiate all the elements in the array.
        var count = Int32(x.count)
        vvexpf(&x, x, &count)
        
        // Compute the sum of all exponentiated values.
        var sum: Float = 0
        vDSP_sve(x, 1, &sum, len)
        
        // Divide each element by the sum. This normalizes the array contents
        // so that they all add up to 1.
        vDSP_vsdiv(x, 1, &sum, &x, 1, len)
        
        let y:[Double] = x.flatMap{Double($0)}
        return y
    }
    
    enum EXIFOrientation : Int32 {
        case topLeft = 1
        case topRight
        case bottomRight
        case bottomLeft
        case leftTop
        case rightTop
        case rightBottom
        case leftBottom
        
        var isReflect:Bool {
            switch self {
            case .topLeft,.bottomRight,.rightTop,.leftBottom: return false
            default: return true
            }
        }
    }
    
    func compensatingEXIFOrientation(deviceOrientation:UIDeviceOrientation) -> EXIFOrientation
    {
        switch (deviceOrientation) {
        case (.landscapeRight): return .bottomRight
        case (.landscapeLeft): return .topLeft
        case (.portrait): return .rightTop
        case (.portraitUpsideDown): return .leftBottom
            
        case (.faceUp): return .rightTop
        case (.faceDown): return .rightTop
        case (_): fallthrough
        default:
            NSLog("Called in unrecognized orientation")
            return .rightTop
        }
    }
}

