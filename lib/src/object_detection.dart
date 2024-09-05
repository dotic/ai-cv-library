import 'dart:developer';
import 'dart:typed_data';
import 'package:ai_cv_library/src/image_analysis_result.dart';
import 'package:ai_cv_library/src/image_processing.dart';
import 'package:ai_cv_library/src/model_loader.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;

class ObjectDetection {
  final ModelLoader _modelLoader = ModelLoader();

  // Analyzes an image to detect objects and returns the results
  Future<ImageAnalysisResult> analyseImage(String imagePath, String yoloPath, String labelsRaw,
      String modelOnnxDetPath, String modelOnnxRecPath, String contentsDict) async {
    int totalPredictionTimeStart = DateTime.now().millisecondsSinceEpoch;

    // Init Yolo model
    print("Initializing yolo model ...");
    await initializeModel(yoloPath, labelsRaw);
    print("ok init yolo");

    // Load and prepare the image for object detection
    print("loading image ...");
    final img.Image imageInput = ImageProcessing.loadImage(imagePath);
    print("ok image loaded");

    // Perform object detection on the image
    print("predict processing ...");
    final predictions = await _modelLoader.processPredict(
        imageInput, modelOnnxDetPath, modelOnnxRecPath, contentsDict);
    print("ok predict");

    // Calculate the total prediction time
    int totalPredictionTimeMs =
        (DateTime.now().millisecondsSinceEpoch - totalPredictionTimeStart).toInt();
    log('Total prediction time: $totalPredictionTimeMs ms');

    // Return results
    Uint8List processedImage = img.encodeJpg(imageInput);
    return ImageAnalysisResult(
      image: processedImage,
      predictions: predictions,
      totalPredictionTimeMs: totalPredictionTimeMs,
    );
  }

  // Initializes the YOLO model and loads labels
  Future<void> initializeModel(String yoloPath, String labelsRaw) async {
    // Load yolo model and labels
    await _modelLoader.loadYoloModel(yoloPath);
    await _modelLoader.loadYoloLabels(labelsRaw);
    log('Model initialization complete.');
  }
}
