import 'dart:convert';
import 'dart:isolate';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';

import 'package:ai_cv_library/src/model_loader.dart';
import 'package:ai_cv_library/src/object_detection.dart';
import 'package:ai_cv_library/src/utils.dart';

class AIComputerVision {
  bool get isAwesome => true;

  Uint8List? image;
  final imagePicker = ImagePicker();
  ObjectDetection? objectDetection;
  List<dynamic>? predictionResults;
  int? totalPredictionTimeMs;

  Future<void> downloadModels(String yoloModelVersion) async {
    // Load s3 credentials
    String s3Cred = await rootBundle.loadString('assets/credentials.json');
    Map<String, dynamic> jsonS3Cred = json.decode(s3Cred);
    // Download models
    return await ModelLoader.downloadFileFromS3(jsonS3Cred, yoloModelVersion);
  }

  String formatPredictions(List<dynamic> predictions) {
    // Initialize an empty string to build the final result
    String formattedPredictions = "";
    for (int i = 0; i < predictions.length; i++) {
      var prediction = predictions[i];
      // Add a line break before displaying an unrecognized class or element, except for the first element
      if (i > 0 &&
          (prediction.containsKey('cls') ||
              !prediction.containsKey('textEtiquette') &&
                  !prediction.containsKey('idPbo'))) {
        formattedPredictions += "\n\n";
      }
      if (prediction.containsKey('textEtiquette')) {
        // Add ocr results
        formattedPredictions +=
            "\nTexte: ${prediction['textEtiquette'].join(", ")}\nConfiance (ocr): ${prediction['confidenceList'].join(", ")}";
      } else if (prediction.containsKey('idPbo')) {
        // Add pbo height
        formattedPredictions +=
            "\nHauteur (Approximative): ${prediction['approximateDistanceFromGroundCm']} cm";
      } else if (prediction.containsKey('cls')) {
        formattedPredictions +=
            "Catégorie : ${prediction['cls']}\nConfiance : ${prediction['score']}";
      } else {
        formattedPredictions += "Non reconnu";
      }
    }
    return formattedPredictions;
  }

  List<String> listModelsVersions() {
    return Utils.yoloModelList;
  }

  void pickAndProcessImage(ImageSource source) async {
    final result = await imagePicker.pickImage(source: source);

    if (result == null) {
      return;
    }

    // Create a port to receive data from the isolate
    final receivePort = ReceivePort();

    // Preload paths and data for image analysis
    final yoloPath = await Utils.getModelPath(
        Utils.modelYoloName); // Preloading the yolo model path
    final labelsRaw =
        await rootBundle.loadString(Utils.labelPath); // Preload labels
    final String modelOnnxDetPath = await Utils.getModelPath(
        Utils.ocrModelOnnxDet); // Preload OCR detection model path
    final String modelOnnxRecPath = await Utils.getModelPath(
        Utils.ocrModelOnnxRec); // Preload OCR recognition model path
    final String contentsDict = await rootBundle
        .loadString(Utils.characterDictPath); // Preload OCR dictionary

    // Launch isolate for image processing
    await Isolate.spawn(_processImageInBackground, [
      receivePort.sendPort,
      result.path,
      yoloPath,
      labelsRaw,
      modelOnnxDetPath,
      modelOnnxRecPath,
      contentsDict
    ]);

    // Listen to the isolate results
    receivePort.listen((data) {
      image = data['image'];
      predictionResults = data['predictions'];
      totalPredictionTimeMs = data['totalTime'];

      // Close the ReceivePort once the data has been received
      receivePort.close();
    });
  }

  static void _processImageInBackground(List<dynamic> args) {
    SendPort sendPort = args[0];
    String path = args[1];
    String yoloPath = args[2];
    String labelsRaw = args[3];
    String modelOnnxDetPath = args[4];
    String modelOnnxRecPath = args[5];
    String contentsDict = args[6];
    // Init ObjectDetection in isolate and process image
    ObjectDetection detection = ObjectDetection();
    _asyncImageProcessing(path, sendPort, detection, yoloPath, labelsRaw,
        modelOnnxDetPath, modelOnnxRecPath, contentsDict);
  }

  static void _asyncImageProcessing(
      String path,
      SendPort sendPort,
      ObjectDetection detection,
      String yoloPath,
      String labelsRaw,
      String modelOnnxDetPath,
      String modelOnnxRecPath,
      String contentsDict) async {
    final results = await detection.analyseImage(path, yoloPath, labelsRaw,
        modelOnnxDetPath, modelOnnxRecPath, contentsDict);
    // Send image analysis results to main port
    sendPort.send({
      'image': results.image,
      'predictions': results.predictions,
      'totalTime': results.totalPredictionTimeMs,
    });
  }
}
