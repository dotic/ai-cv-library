import 'dart:developer';
import 'dart:typed_data';
import 'dart:math' as math;
import 'package:ai_cv_library/src/image_processing.dart';
import 'package:ai_cv_library/src/utils.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:image/image.dart' as img;
import 'package:tuple/tuple.dart';
import 'native_opencv.dart' as cv2;
import 'dart:io';
import 'package:aws_client/s3_2006_03_01.dart';
import 'package:path_provider/path_provider.dart';

class ModelLoader {
  Interpreter? _interpreter;
  List<String>? _labels;
  List<dynamic>? _predictions;

  // Download models from s3 bucket
  static Future<void> downloadFileFromS3(Map<String, dynamic> config, String yoloModelVersion) async {
    try {
      final credentials = AwsClientCredentials(accessKey: config['awsAccessKey'], secretKey: config['awsSecretKey']);
      final s3 = S3(region: config['awsRegion'], credentials: credentials);

      String modelYoloName = "${(Utils.modelYoloName).replaceAll('.tflite', '')}_$yoloModelVersion.tflite";
      List<String> modelList = [Utils.ocrModelOnnxDet, Utils.ocrModelOnnxRec, modelYoloName];
      for (String filePath in modelList) {
        //final directory = (await getExternalStorageDirectory())!;
        final directory = await getApplicationDocumentsDirectory();

        // set model name without the version
        RegExp regExp = RegExp(r'model.*\.tflite');
        String updatedFilePath = filePath.replaceAll(regExp, 'model.tflite');
        final file = File('${directory.path}/$updatedFilePath');

        final response = await s3.headObject(bucket: config['bucketName'], key: filePath);
        final s3LastModified = response.lastModified;

        if (s3LastModified == null) {
          throw Exception("Unable to retrieve last modification date of S3 file");
        }

        // Check if the local file exists and compare modification dates
        if (await file.exists()) {
          final localFileModified = await file.lastModified();
          if (localFileModified.isAtSameMomentAs(s3LastModified)) {
            log('The local file is up to date : $filePath');
            continue;
          } else {
            log('New version of $filePath available');
          }
        }

        // Download file if local file does not exist or is obsolete
        log("Downloading $filePath ...");
        final getObjectResponse = await s3.getObject(bucket: config['bucketName'], key: filePath);
        if (getObjectResponse.body == null) {
          throw Exception("No content found in answer S3");
        }
        final bytes = getObjectResponse.body;
        await file.writeAsBytes(bytes!);
        // Update date of last modification
        await file.setLastModified(s3LastModified);
        log('File successfully downloaded : ${file.path}');
      }
    } catch (e) {
      log('Error checking file: $e');
    }
  }

  // Load Yolo model
  Future<void> loadYoloModel(String yoloPath) async {
    log('Loading interpreter options...');
    final interpreterOptions = InterpreterOptions();
    log('Loading interpreter...');
    final File fileYolo = File(yoloPath);
    print('fileYolo : ${fileYolo.toString()}');
    print("interpreter : $_interpreter");
    if (_interpreter != null) {
      _interpreter?.close();
    }

    if (Platform.isAndroid) {
      interpreterOptions.addDelegate(XNNPackDelegate());
    }

    if (Platform.isIOS) {
      interpreterOptions.addDelegate(GpuDelegate());
    }

    _interpreter = Interpreter.fromFile(fileYolo, options: interpreterOptions);
    print('_interpreter initialized: ${_interpreter != null}');
  }

  // Load labels
  Future<void> loadYoloLabels(String labelsRaw) async {
    log('Formating labels...');
    _labels = labelsRaw.split('\n');
  }

  // Perform predictions using an ONNX model
  static dynamic onnxPred(List data, List<int> shape, String modelPath, String inputDictKey) async {
    OrtEnv.instance.init();
    final sessionOptions = OrtSessionOptions();
    final modelFile = File(modelPath);
    final bytes = await modelFile.readAsBytes();
    final session = OrtSession.fromBuffer(bytes, sessionOptions);
    final runOptions = OrtRunOptions();
    final inputOrt = OrtValueTensor.createTensorWithDataList(data, shape);
    final inputs = {inputDictKey: inputOrt};
    final outputs = session.run(runOptions, inputs);
    inputOrt.release();
    runOptions.release();
    sessionOptions.release();
    OrtEnv.instance.release();
    return outputs;
  }

  // Format output
  List<dynamic> processOutputs(List<dynamic> output) {
    final results = <dynamic>[];

    for (var i = 0; i < output.length; i++) {
      var x1 = output[i][1].toInt();
      var y1 = output[i][2].toInt();
      var x2 = output[i][3].toInt();
      var y2 = output[i][4].toInt();
      var clsId = output[i][5].toInt();
      var cls = _labels![clsId];
      var score = output[i][6];

      if (score >= Utils.scoreThreshold) {
        results.add({
          'x1': x1,
          'y1': y1,
          'x2': x2,
          'y2': y2,
          'score': double.parse(score.toStringAsFixed(3)),
          'cls': cls.substring(0, cls.length - 1)
        });
      }
    }
    return results;
  }

  Future<List<dynamic>?> processPredict(
      img.Image imageInput, String modelOnnxDetPath, String modelOnnxRecPath, String contentsDict) async {
    //init prediction
    _predictions = [];

    //Convert image to matrix
    print("converting to matrix ...");
    final List<List<List<num>>> convertedImageToMatrix = ImageProcessing.convertImageToMatrix(imageInput);
    print("ok img to matrix");

    // Convert image from channel last format to channel first format
    print("converting to channel first format ...");
    final List<List<List<num>>> input = ImageProcessing.convertChannelsLastToChannelsFirst(convertedImageToMatrix);
    print("ok convert channel first");

    //Yolo prediction
    //tflite prediction
    final output = List<num>.filled(100 * 7, 0).reshape([100, 7]);

    print("running interpreter ...");
    print("input : $input");
    print("output : $output");
    try {
      _interpreter?.run([input], output);
      log('Interpreter ran successfully.');
    } catch (e) {
      log('Error during model inference: $e');
    }
    print("ok run interpreter");

    final results = processOutputs(output);

    for (var r in results) {
      //add to global prediction list
      _predictions!.add(r);

      print("-----");
      print(r);

      // if tag "pbo" is detected and ref is true
      if (r['cls'].toString().contains('pbo')) {
        await _processPboTag(r, imageInput, results);
      }
      // if tags "plaque_immatriculation" or "visage" are detected
      if (r['cls'].toString().contains('plaque_immatriculation') || r['cls'].toString().contains('visage')) {
        _processPlateOrFaceLabel(r, imageInput);
      }
      // if tag "etiquette" is detected
      if (r['cls'].toString().contains('etiquette')) {
        await _processTagLabel(r, imageInput, input, modelOnnxDetPath, modelOnnxRecPath, contentsDict);
      }

      // Draw results bboxes
      Utils.drawRectangle(imageInput, r);
      String toWrite = '${r["cls"]} ${r["score"]}';
      Utils.drawLabel(imageInput, toWrite, r["x1"] + 4, r["y1"] + 4);
    }

    log('Done.');

    _interpreter?.close();

    return _predictions;
  }

  Future<void> _processPboTag(Map<String, dynamic> r, img.Image imageInput, var results) async {
    bool containsRef = results.any((r) => ['facade', 'poteau', 'tuyau'].contains(r['cls'].toString()));
    if (containsRef) {
      var dfRef = results.where((r) => ['facade', 'poteau', 'tuyau'].contains(r['cls'].toString())).toList();
      String idPbo = "pbo_x1:${r['x1']}_y1:${r['y1']}_x2:${r['x2']}_y2:${r['y2']}";
      double minDistance = double.maxFinite;
      Map<String, dynamic>? refToCheck;

      // find the closest ref to each pbo then check the distance from the ground (both bottom)
      for (var dataRef in dfRef) {
        double dLeft = (r['x1'] - dataRef['x1']).abs().toDouble();
        double dRight = (r['x2'] - dataRef['x2']).abs().toDouble();
        double minDistanceActualRef = dLeft > dRight ? dRight : dLeft;
        if (minDistanceActualRef < minDistance) {
          minDistance = minDistanceActualRef;
          refToCheck = dataRef;
        }
      }
      // Calculate the height of the 'pbo' from the ground if a reference object is found
      if (refToCheck != null) {
        double pSizePbo = (r['y2'] - r['y1']).toDouble();
        double pDistancePbo = (refToCheck['y2'] - r['y2']).toDouble();
        double hCmPbo = 25;
        int cmDistancePbo = ((pDistancePbo / pSizePbo) * hCmPbo).toInt();
        log("Height pbo : $cmDistancePbo cm (id pbo = $idPbo)");

        // Add to global prediction list
        var dicResPbo = {"idPbo": idPbo, "approximateDistanceFromGroundCm": cmDistancePbo};
        _predictions!.add(dicResPbo);

        // Prediction height drawing
        String toWrite = 'Height pbo : $cmDistancePbo cm';
        Utils.drawLabel(imageInput, toWrite, r["x1"] + 4, r["y1"] + 14);
      } else {
        log("No reference object found for pbo");
      }
    } else {
      log("Can't calculate height");
    }
  }

  void _processPlateOrFaceLabel(Map<String, dynamic> r, img.Image imageInput) {
    log("Blurring in progress ...");

    int x1 = r["x1"].toInt();
    int y1 = r["y1"].toInt();
    int x2 = r["x2"].toInt();
    int y2 = r["y2"].toInt();

    // Adjust dimensions to stay within image limits
    int cropWidth = math.min(x2 - x1, imageInput.width - x1);
    int cropHeight = math.min(y2 - y1, imageInput.height - y1);

    // Crop the area of the image that needs blurring
    final toBlur = img.copyCrop(imageInput, x: x1, y: y1, width: cropWidth, height: cropHeight);

    // Calculate the blur radius, ensuring it's at least 1
    int radius = (math.min(toBlur.width, toBlur.height) * 0.6).toInt();
    radius = math.max(1, radius);

    // Apply Gaussian blur to the cropped area
    img.gaussianBlur(toBlur, radius: radius);

    // Replace the original area in the image with the blurred area
    for (int y = 0; y < toBlur.height; y++) {
      for (int x = 0; x < toBlur.width; x++) {
        imageInput.setPixel(x1 + x, y1 + y, toBlur.getPixel(x, y));
      }
    }
  }

  Future<void> _processTagLabel(Map<String, dynamic> r, img.Image imageInput, List<List<List<num>>> input,
      String modelOnnxDetPath, String modelOnnxRecPath, String contentsDict) async {
    log("Tag detected");
    log("Reading text ...");

    // Crop and preprocessing for ocr
    Tuple3<List<List<List<int>>>, List<List<List<double>>>, List<double>> imagePreprocessed =
        ImageProcessing.preprocessImageForOcr(imageInput, r);

    // KeepKeys
    List<List<List<int>>> imageInputList = imagePreprocessed.item1;
    List<List<List<List<double>>>> dataImageNorm = [imagePreprocessed.item2];
    List<List<double>> shapeList = [imagePreprocessed.item3];

    // OCR detection
    List<dynamic> detectionResult = await _performOcrDetection(dataImageNorm, shapeList, modelOnnxDetPath);

    // OCR recognition
    List<Tuple2<String, double>> recognitionResult =
        await _performOcrRecognition(input, imageInputList, detectionResult, modelOnnxRecPath, contentsDict);

    // Improve OCR prediction
    List<dynamic> improvedText = Utils.improveTextPrediction(recognitionResult);
    print(improvedText);

    // Add to global prediction list
    if (improvedText[0].isNotEmpty) {
      var etiRes = {
        "textEtiquette": improvedText[0],
        "confidenceList": improvedText[1],
        "idEtiquette": "eti_x1:${r['x1']}_y1:${r['y1']}_x2:${r['x2']}_y2:${r['y2']}"
      };
      _predictions!.add(etiRes);
    }
  }

  Future<List<dynamic>> _performOcrDetection(
      List<List<List<List<double>>>> dataImageNorm, List<List<double>> shapeList, String modelOnnxDetPath) async {
    var outputDet = await _predictWithOnnxDetection(dataImageNorm, modelOnnxDetPath);
    return _processDetectionResults(outputDet, shapeList);
  }

  Future<dynamic> _predictWithOnnxDetection(
      List<List<List<List<double>>>> dataImageNorm, String modelOnnxDetPath) async {
    // onnx detection prediction
    List<int> onnxDetShape = [
      dataImageNorm.shape[0],
      dataImageNorm.shape[1],
      dataImageNorm.shape[2],
      dataImageNorm.shape[3]
    ];
    List<List<List<Float32List>>> inputToFloat = dataImageNorm
        .map((list3D) =>
            list3D.map((list2D) => list2D.map((innerList) => Float32List.fromList(innerList)).toList()).toList())
        .toList();

    String inputDictKey = 'x';
    var outputDet = (await onnxPred(inputToFloat, onnxDetShape, modelOnnxDetPath, inputDictKey))[0].value;

    return outputDet;
  }

  List<dynamic> _processDetectionResults(dynamic outputDet, List<List<double>> shapeList) {
    // Results post process
    List<List<List<double>>> pred = List<List<List<double>>>.from(
      outputDet[0].map(
        (item) => List<List<double>>.from(
          item.map(
            (innerItem) => List<double>.from(innerItem),
          ),
        ),
      ),
    );

    // Create segmentation
    List<List<List<bool>>> segmentation = _createSegmentation(pred);

    List<dynamic> boxesBatch = [];
    for (int batchIndex = 0; batchIndex < pred.shape[0]; batchIndex++) {
      int srcH = shapeList[0][0].toInt();
      int srcW = shapeList[0][1].toInt();
      List<List<bool>> mask = segmentation[batchIndex];

      int height = mask.shape[0];
      int width = mask.shape[1];
      List<List<double>> predBatchIndex = pred[batchIndex];

      // Find contours in the mask
      Uint8List maskData = ImageProcessing.convertBoolMaskToUint8List(mask);
      List<List<List<int>>> contours = cv2.findContoursInMask(maskData, width, height);
      int numContours = math.min(contours.length, Utils.maxCandidates);

      // Process each contour to create boxes
      List boxes = [];
      List scores = [];
      for (int index = 0; index < numContours; index++) {
        List<List<int>> contour = contours[index];
        var rGetMiniBox = Utils.getMiniBox(contour);
        var box = rGetMiniBox['box'];
        var sside = rGetMiniBox['sside'];

        // Skip small contours
        if (sside < Utils.minSize) continue;

        List<List<double>> pointsArray = box.map<List<double>>((Tuple2<double, double> tuple) {
          return [tuple.item1, tuple.item2];
        }).toList();
        List<List<double>> box0 = List.from(pointsArray);

        int h = predBatchIndex.length;
        int w = predBatchIndex[0].length;
        // Calculation of xmin, xmax, ymin, ymax (bounding box dimensions)
        double xmin = box0.map((e) => e[0]).reduce(math.min).floor().clamp(0, w - 1).toDouble();
        double xmax = box0.map((e) => e[0]).reduce(math.max).ceil().clamp(0, w - 1).toDouble();
        double ymin = box0.map((e) => e[1]).reduce(math.min).floor().clamp(0, h - 1).toDouble();
        double ymax = box0.map((e) => e[1]).reduce(math.max).ceil().clamp(0, h - 1).toDouble();

        // Create and fill a mask for the box
        int maskHeight = (ymax - ymin + 1).toInt();
        int maskWidth = (xmax - xmin + 1).toInt();
        List<List<int>> mask = List.generate(maskHeight, (_) => List.filled(maskWidth, 0));
        // Adjust box coordinates
        for (var point in box0) {
          point[0] -= xmin;
          point[1] -= ymin;
        }
        List<List<int>> modifiedMask = cv2.fillPoly(mask, box0);

        // Calculate the score for the box
        double score = 0.0;
        double count = 0;
        for (int i = ymin.toInt(); i <= ymax.toInt(); i++) {
          for (int j = xmin.toInt(); j <= xmax.toInt(); j++) {
            if (modifiedMask[i - ymin.toInt()][j - xmin.toInt()] == 1) {
              score += predBatchIndex[i][j];
              count += 1;
            }
          }
        }
        score = count != 0 ? score / count : 0.0;

        // Skip boxes below threshold score
        if (Utils.boxThresh > score) continue;

        // Expand and recompute the box based on the area and perimeter
        double polyArea = Utils.polygonArea(box);
        double polyLength = Utils.polygonPerimeter(box);
        double distance = (polyArea * Utils.unclipRatio) / polyLength;
        List<List<int>> expandedBox = Utils.expandRectangle(box, distance);

        // Finalize the box and add it to the batch
        rGetMiniBox = Utils.getMiniBox(expandedBox);
        box = rGetMiniBox['box'];
        sside = rGetMiniBox['sside'];
        if (sside < Utils.minSize + 2) continue;

        pointsArray = box.map<List<double>>((Tuple2<double, double> tuple) {
          return [tuple.item1, tuple.item2];
        }).toList();

        boxes.add(Utils.scaleAndClipBox(pointsArray, width, srcH, srcW, height));
        scores.add(score);
      }
      boxesBatch.add({'points': boxes});
    }

    return boxesBatch;
  }

  List<List<List<bool>>> _createSegmentation(List<List<List<double>>> pred) {
    List<List<List<bool>>> segmentation = [];
    for (int i = 0; i < 1; i++) {
      List<List<bool>> row = [];
      // Iterate through each row in the prediction
      for (int j = 0; j < pred[0].length; j++) {
        List<bool> col = [];
        // Iterate through each column in the row
        for (int k = 0; k < pred[0][j].length; k++) {
          col.add(pred[0][j][k] > Utils.detThresh);
        }
        row.add(col);
      }
      segmentation.add(row);
    }
    return segmentation;
  }

  Future<List<Tuple2<String, double>>> _performOcrRecognition(
      List<List<List<num>>> input,
      List<List<List<int>>> imageInputList,
      List<dynamic> boxesBatch,
      String modelOnnxRecPath,
      String contentsDict) async {
    var imgH = input.shape[1];
    var imgW = input.shape[2];

    // Prepare Images For Recognition
    List<List<List<int>>> dtBoxes = _calculateDtBoxes(boxesBatch, imgH, imgW);
    List<List<List<List<double>>>> imgCropList = _createImgCropList(dtBoxes, imageInputList);

    // Text recognize
    var imgNum = imgCropList.length;
    List<Tuple2<String, double>> recRes = List.generate(imgNum, (_) => const Tuple2('', 0.0));

    //Calculate the aspect ratio of all text bars
    List<double> widthList = [];
    for (var img in imgCropList) {
      widthList.add(img.shape[1] / img.shape[0]);
    }

    //Sorting to speed up the recognition process
    List<int> indices = Utils.argsort(widthList);

    int batchNum = Utils.recBatchNum;
    for (int begImgNo = 0; begImgNo < imgNum; begImgNo += batchNum) {
      int endImgNo = math.min(imgNum, begImgNo + batchNum);

      // Normalize image
      List<List<List<List<double>>>> normImgBatch = _normalizeForOcr(imgCropList, indices, begImgNo, endImgNo);

      // Onnx recognition prediction
      List<List<List<double>>> preds = await _predictWithOnnxRecognition(normImgBatch, modelOnnxRecPath);

      // Process and decode Ocr results
      List<Tuple2<String, double>> recResult = await _decodeOcrResults(preds, contentsDict);
      for (int rno = 0; rno < recResult.length; rno++) {
        recRes[indices[begImgNo + rno]] = recResult[rno];
      }
    }

    List<Tuple2<String, double>> filterRecRes = _processFilterRecognitionResults(recRes, dtBoxes);
    print(filterRecRes);

    return filterRecRes;
  }

  List<List<List<int>>> _calculateDtBoxes(List<dynamic> boxesBatch, int imgH, int imgW) {
    List<List<List<int>>> dtBoxesNew = [];
    var dtBoxes = boxesBatch[0]['points'];
    for (var pts in dtBoxes) {
      var xSorted = List<List<int>>.from(pts)..sort(Utils.compareFunction);
      // The two points furthest to the left and right
      var leftMost = xSorted.sublist(0, 2);
      var rightMost = xSorted.sublist(2);
      // Sort the leftmost points by y to obtain the topmost and bottommost points.
      leftMost.sort((List<int> a, List<int> b) => a[1].compareTo(b[1]));
      var tl = leftMost[0];
      var bl = leftMost[1];
      // The same applies to the rightmost points
      rightMost.sort((List<int> a, List<int> b) => a[1].compareTo(b[1]));
      var tr = rightMost[0];
      var br = rightMost[1];
      // Combine all points in the correct order
      var box = [tl, tr, br, bl];

      for (int pno = 0; pno < box.length; pno++) {
        box[pno][0] = box[pno][0].clamp(0, imgW - 1);
        box[pno][1] = box[pno][1].clamp(0, imgH - 1);
      }

      int rectWidth = (Utils.euclideanDistance(box[0], box[1])).toInt();
      int rectHeight = (Utils.euclideanDistance(box[0], box[3])).toInt();

      if (rectWidth > 3 && rectHeight > 3) {
        dtBoxesNew.add(box);
      }
    }
    dtBoxes = List<List<List<int>>>.from(dtBoxesNew);

    int numBoxes = dtBoxes.length;
    // Sort boxes by y, then x
    List<List<List<int>>> sortedBoxes = dtBoxes;
    sortedBoxes.sort((a, b) {
      int comparison = a[0][1].compareTo(b[0][1]);
      if (comparison == 0) {
        return a[0][0].compareTo(b[0][0]);
      }
      return comparison;
    });
    List<List<List<int>>> boxes0 = List<List<List<int>>>.from(sortedBoxes);
    // Checks and adjusts boxes
    for (int i = 0; i < numBoxes - 1; i++) {
      if (((boxes0[i + 1][0][1] - boxes0[i][0][1]).abs() < 10) && (boxes0[i + 1][0][0] < boxes0[i][0][0])) {
        var tmp = boxes0[i];
        boxes0[i] = boxes0[i + 1];
        boxes0[i + 1] = tmp;
      }
    }
    return boxes0; //dtBoxes = boxes0
  }

  // Function to create a list of cropped images from detected boxes
  List<List<List<List<double>>>> _createImgCropList(
      List<List<List<int>>> dtBoxes, List<List<List<int>>> imageInputList) {
    List<List<List<List<double>>>> imgCropList = [];
    for (int bno = 0; bno < dtBoxes.length; bno++) {
      List<List<int>> tmpBox = List.from(dtBoxes[bno]);

      // Ensure the shape of points is correct
      assert(tmpBox.length == 4, "shape of points must be 4*2");

      // Calculate distances to determine the width and height of the crop
      double distance01 = Utils.euclideanDistance(tmpBox[0], tmpBox[1]);
      double distance23 = Utils.euclideanDistance(tmpBox[2], tmpBox[3]);
      int imgCropWidth = distance01 > distance23 ? distance01.toInt() : distance23.toInt();
      double distance03 = Utils.euclideanDistance(tmpBox[0], tmpBox[3]);
      double distance12 = Utils.euclideanDistance(tmpBox[1], tmpBox[2]);
      int imgCropHeight = distance03 > distance12 ? distance03.toInt() : distance12.toInt();

      // Standard points for the perspective transformation
      List<List<int>> ptsStd = [
        [0, 0],
        [imgCropWidth, 0],
        [imgCropWidth, imgCropHeight],
        [0, imgCropHeight]
      ];

      // Get perspective transformation matrix
      List<List<double>> M = cv2.callGetPerspectiveTransform(tmpBox, ptsStd);

      // Apply warp perspective
      List<List<List<int>>> srcImageDataTemp = imageInputList
          .map((list2D) => list2D.map((list1D) => list1D.map((value) => (value).toInt()).toList()).toList())
          .toList();

      Uint8List srcImageData = ImageProcessing.convertNestedListToUint8List(srcImageDataTemp);
      int srcWidth = srcImageDataTemp[0].length;
      int srcHeight = srcImageDataTemp.length;
      List<List<List<double>>> dstImg =
          cv2.warpPerspective(srcImageData, srcWidth, srcHeight, M, imgCropWidth, imgCropHeight);

      // Rotate image if the height/width ratio is greater than 1.5
      int dstImgHeight = dstImg.length;
      int dstImgWidth = dstImg[0].length;
      if (dstImgHeight * 1.0 / dstImgWidth >= 1.5) {
        dstImg = ImageProcessing.rotate90DegreesLeft(dstImg);
      }

      imgCropList.add(dstImg);
    }
    return imgCropList;
  }

  List<List<List<List<double>>>> _normalizeForOcr(
      List<List<List<List<double>>>> imgCropList, List<int> indices, int begImgNo, int endImgNo) {
    List<List<List<List<double>>>> normImgBatch = [];
    double maxWhRatio = 0;

    // Find the maximum width-to-height ratio among the cropped images
    for (int ino = begImgNo; ino < endImgNo; ino++) {
      int h = imgCropList[indices[ino]].length;
      int w = imgCropList[indices[ino]][0].length;
      double whRatio = w * 1.0 / h;
      maxWhRatio = math.max(maxWhRatio, whRatio);
    }

    // Normalize each image in the specified range
    for (int ino = begImgNo; ino < endImgNo; ino++) {
      int imgC = Utils.recImageShape[0];
      int imgH = Utils.recImageShape[1];
      int imgW = Utils.recImageShape[2];
      assert(imgC == imgCropList[indices[ino]][0][0].length);

      // Adjust the width of the image according to the max width-to-height ratio
      imgW = (32 * maxWhRatio).toInt();
      double h = imgCropList[indices[ino]].length.toDouble();
      double w = imgCropList[indices[ino]][0].length.toDouble();
      double ratio = w / h;
      int resizedW;
      if ((imgH * ratio).ceil() > imgW) {
        resizedW = imgW;
      } else {
        resizedW = (imgH * ratio).ceil();
      }

      // Resize image for OCR input
      img.Image originalImage = ImageProcessing.imgCropListToImageObject(imgCropList[indices[ino]]);
      img.Image resizedImageObj = img.copyResize(originalImage, width: resizedW, height: imgH);
      List<List<List<double>>> resizedImage = ImageProcessing.imageObjectToImgCropList(resizedImageObj);

      // Normalize pixel values
      for (int i = 0; i < resizedImage.length; i++) {
        for (int j = 0; j < resizedImage[i].length; j++) {
          for (int k = 0; k < resizedImage[i][j].length; k++) {
            resizedImage[i][j][k] = (resizedImage[i][j][k] / 255.0 - 0.5) / 0.5;
          }
        }
      }
      resizedImage = ImageProcessing.convertChannelsLastToChannelsFirst(resizedImage)
          .map((list2D) => list2D.map((list) => list.map((item) => item.toDouble()).toList()).toList())
          .toList();

      // Create a normalized image batch for OCR
      List<List<List<double>>> normImg = List.generate(
          imgC, (c) => List.generate(imgH, (h) => List.generate(imgW, (w) => 0.0, growable: false), growable: false),
          growable: false);
      for (int c = 0; c < imgC; c++) {
        for (int h = 0; h < imgH; h++) {
          for (int w = 0; w < resizedW; w++) {
            normImg[c][h][w] = resizedImage[c][h][w];
          }
        }
      }

      normImgBatch.add(normImg);
    }
    return normImgBatch;
  }

  // Onnx recognition prediction
  Future<List<List<List<double>>>> _predictWithOnnxRecognition(
      List<List<List<List<double>>>> normImgBatch, String modelOnnxRecPath) async {
    List<int> onnxRecShape = [
      normImgBatch.shape[0],
      normImgBatch.shape[1],
      normImgBatch.shape[2],
      normImgBatch.shape[3]
    ];
    List<List<List<Float32List>>> inputToFloat = normImgBatch
        .map((list3D) =>
            list3D.map((list2D) => list2D.map((innerList) => Float32List.fromList(innerList)).toList()).toList())
        .toList();

    String inputDictKey = 'x';
    List<List<List<double>>> outputRec =
        (await onnxPred(inputToFloat, onnxRecShape, modelOnnxRecPath, inputDictKey))[0].value;
    return outputRec;
  }

  // Decode OCR results into human-readable text
  Future<List<Tuple2<String, double>>> _decodeOcrResults(List<List<List<double>>> preds, String contents) async {
    // Load and prepare the character dictionary
    List<String> lines = contents.split('\n');
    List<String> characters = lines.map((line) => line.trim()).toList();
    if (Utils.useSpaceChar) {
      characters.insert(0, "blank");
    }
    Map<String, int> dictCharacters = {};
    for (int i = 0; i < characters.length; i++) {
      dictCharacters[characters[i]] = i;
    }

    // Prepare lists to hold text indices and probabilities
    List<List<int>> textIndex = [];
    List<List<double>> textProbs = [];
    for (var matrix in preds) {
      List<int> matrixIdx = [];
      List<double> matrixProb = [];
      for (var sublist in matrix) {
        int maxIdx = sublist.indexWhere((x) => x == sublist.map((e) => e.toDouble()).reduce(math.max));
        matrixIdx.add(maxIdx);
        matrixProb.add(sublist[maxIdx]);
      }
      textIndex.add(matrixIdx);
      textProbs.add(matrixProb);
    }

    // Decode results into readable text
    List<Tuple2<String, double>> recResult = [];
    List<int> ignoredTokens = [0];
    int batchSize = textIndex.length;
    for (var batchIdx = 0; batchIdx < batchSize; batchIdx++) {
      var charList = <String>[];
      var confList = <double>[];
      for (var idx = 0; idx < textIndex[batchIdx].length; idx++) {
        if (ignoredTokens.contains(textIndex[batchIdx][idx])) continue;
        if (idx > 0 && textIndex[batchIdx][idx - 1] == textIndex[batchIdx][idx]) {
          continue;
        }

        charList.add(characters[textIndex[batchIdx][idx]]);
        confList.add(textProbs[batchIdx][idx]);
      }
      String text = charList.join();
      double confidence = confList.isNotEmpty ? confList.reduce((a, b) => a + b) / confList.length : 0.0; // = np.mean
      recResult.add(Tuple2<String, double>(text, confidence));
    }
    return recResult;
  }

  // Filter OCR recognition results based on a confidence threshold
  List<Tuple2<String, double>> _processFilterRecognitionResults(
      List<Tuple2<String, double>> recognitionResults, var dtBoxes) {
    List<dynamic> filterBoxes = [];
    List<Tuple2<String, double>> filterRecRes = [];
    for (int i = 0; i < dtBoxes.length; i++) {
      var box = dtBoxes[i];
      double score = recognitionResults[i].item2;
      if (score >= Utils.dropScore) {
        filterBoxes.add(box);
        filterRecRes.add(recognitionResults[i]);
      }
    }
    return filterRecRes;
  }
}
