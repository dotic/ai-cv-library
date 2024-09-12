import 'dart:convert';
import 'dart:developer';
import 'dart:io';
import 'dart:isolate';
import 'package:flutter/cupertino.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';

import 'package:ai_cv_library/src/model_loader.dart';
import 'package:ai_cv_library/src/object_detection.dart';
import 'package:ai_cv_library/src/utils.dart';

import 'image_analysis_result.dart';

class AIComputerVision {
  bool get isAwesome => true;

  late Uint8List image;
  final ImagePicker imagePicker = ImagePicker();
  ObjectDetection? objectDetection;
  List<dynamic>? predictionResults;
  int? totalPredictionTimeMs;

  Future<bool> checkAvailableModels() async {
    final String yoloModelPath = await Utils.getModelPath(Utils.modelYoloName);
    final String ocrDetModelPath = await Utils.getModelPath(Utils.ocrModelOnnxDet);
    final String ocrRecModelPath = await Utils.getModelPath(Utils.ocrModelOnnxRec);

    if (File(yoloModelPath).existsSync() && File(ocrDetModelPath).existsSync() && File(ocrRecModelPath).existsSync()) {
      return true;
    } else {
      log('Unable to load all needed models. Try downloading them.');
      return false;
    }
  }

  Future<void> downloadModels(String yoloModelVersion) async {
    // Load s3 credentials
    final String s3Cred = await rootBundle.loadString('assets/aws/credentials.json');
    final Map<String, dynamic> jsonS3Cred = json.decode(s3Cred) as Map<String, dynamic>;
    // Download models
    return ModelLoader.downloadFileFromS3(jsonS3Cred, yoloModelVersion);
  }

  String formatPredictions(List<dynamic> predictions) {
    // Initialize an empty string to build the final result
    String formattedPredictions = '';
    for (int i = 0; i < predictions.length; i++) {
      final dynamic prediction = predictions[i];
      // Add a line break before displaying an unrecognized class or element, except for the first element
      if (i > 0 &&
          (prediction.containsKey('cls') != null ||
              prediction.containsKey('textEtiquette') == null && prediction.containsKey('idPbo') == null)) {
        formattedPredictions += "\n\n";
      }
      if (prediction.containsKey('textEtiquette') != null) {
        // Add ocr results
        formattedPredictions +=
            "\nTexte: ${prediction['textEtiquette'].join(", ")}\nConfiance (ocr): ${prediction['confidenceList'].join(", ")}";
      } else if (prediction.containsKey('idPbo') != null) {
        // Add pbo height
        formattedPredictions += "\nHauteur (Approximative): ${prediction['approximateDistanceFromGroundCm']} cm";
      } else if (prediction.containsKey('cls') != null) {
        formattedPredictions += "Catégorie : ${prediction['cls']}\nConfiance : ${prediction['score']}";
      } else {
        formattedPredictions += "Non reconnu";
      }
    }
    return formattedPredictions;
  }

  List<String> listModelsVersions() {
    return Utils.yoloModelList;
  }

  Future<ImageAnalysisResult> pickAndProcessImage(
    ImageSource source, {
    String? imagePath,
  }) async {
<<<<<<< debug-ios
=======
    print("1!!!!!");

    print("source(api) : ${source.toString()}");
    print("imagePath(api) : ${imagePath.toString()}");

>>>>>>> main
    final String? path = imagePath ?? (await imagePicker.pickImage(source: source))?.path;

    if (path == null || path.isEmpty) {
      throw Exception('No image selected');
    }

    // Create a port to receive data from the isolate
    final ReceivePort receivePort = ReceivePort();

    // Preload paths and data for image analysis
    final String yoloPath = await Utils.getModelPath(Utils.modelYoloName); // Preloading the yolo model path
    final String labelsRaw = await rootBundle.loadString(Utils.labelPath); // Preload labels
    final String modelOnnxDetPath = await Utils.getModelPath(Utils.ocrModelOnnxDet); // Preload OCR detection model path
    final String modelOnnxRecPath =
        await Utils.getModelPath(Utils.ocrModelOnnxRec); // Preload OCR recognition model path
    final String contentsDict = await rootBundle.loadString(Utils.characterDictPath); // Preload OCR dictionary

    // Launch isolate for image processing
    await Isolate.spawn(
      _processImageInBackground,
      <Object>[
        receivePort.sendPort,
        path,
        yoloPath,
        labelsRaw,
        modelOnnxDetPath,
        modelOnnxRecPath,
        contentsDict,
      ],
    );

    // Listen and await results from the isolate
    final dynamic data = await receivePort.first;

    if ((data as dynamic)['error'] != null) {
      throw Exception((data as dynamic)['error']);
    }

    image = (data as dynamic)['image'] as Uint8List;
    predictionResults = (data as dynamic)['predictions'] as List<dynamic>;
    totalPredictionTimeMs = (data as dynamic)['totalTime'] as int?;

    print('predictionResults : ${predictionResults.toString()}');
    print('totalPredictionTimeMs : ${totalPredictionTimeMs.toString()}');

    if (predictionResults == null || totalPredictionTimeMs == null) {
      throw Exception('Error processing image');
    }

    return ImageAnalysisResult(
      image: image,
      predictions: predictionResults,
      totalPredictionTimeMs: totalPredictionTimeMs!,
    );
  }

  static void _processImageInBackground(List<dynamic> args) {
    final SendPort sendPort = args[0] as SendPort;
    final String path = args[1] as String;
    final String yoloPath = args[2] as String;
    final String labelsRaw = args[3] as String;
    final String modelOnnxDetPath = args[4] as String;
    final String modelOnnxRecPath = args[5] as String;
    final String contentsDict = args[6] as String;
    // Init ObjectDetection in isolate and process image
    final ObjectDetection detection = ObjectDetection();
    _asyncImageProcessing(
        path, sendPort, detection, yoloPath, labelsRaw, modelOnnxDetPath, modelOnnxRecPath, contentsDict);
  }

  static Future<void> _asyncImageProcessing(
    String path,
    SendPort sendPort,
    ObjectDetection detection,
    String yoloPath,
    String labelsRaw,
    String modelOnnxDetPath,
    String modelOnnxRecPath,
    String contentsDict,
  ) async {
    try {
      final ImageAnalysisResult results =
          await detection.analyseImage(path, yoloPath, labelsRaw, modelOnnxDetPath, modelOnnxRecPath, contentsDict);
      // Send image analysis results to main port
      sendPort.send(<String, Object?>{
        'image': results.image,
        'predictions': results.predictions,
        'totalTime': results.totalPredictionTimeMs,
      });
      debugPrint('Results sent from isolate'); // Debug print
    } catch (e) {
      debugPrint('Error in isolate: $e');
      sendPort.send(<String, String>{
        'error': e.toString(),
      });
    }
  }
}
