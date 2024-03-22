import 'dart:typed_data';

// A class to represent the results of image analysis.
class ImageAnalysisResult {
  Uint8List image;
  List<dynamic>? predictions;
  int totalPredictionTimeMs;

  ImageAnalysisResult({
    required this.image,
    this.predictions,
    required this.totalPredictionTimeMs,
  });
}
