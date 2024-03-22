import 'dart:io';

import 'package:path_provider/path_provider.dart';
import 'package:tuple/tuple.dart';
import 'dart:math' as math;
import 'package:image/image.dart' as img;

class Utils {
  // yolo detection
  static const List<String> yoloModelList = ["v1_0", "v1_1"];
  static const String modelYoloName =
      'yolov7_model.tflite'; //'yolov7_model.tflite.enc';
  static const String modelYoloOnnxPath =
      'yolov7_model.onnx'; //'yolov7_model.onnx.enc';
  static const String labelPath = 'assets/labels.txt';
  static const double scoreThreshold = 0.4;

  //ocr detection
  static const String ocrModelOnnxDet =
      'paddleocr_onnx_det_model.onnx'; //'paddleocr_onnx_det_model.onnx.enc';
  static const String ocrModelOnnxRec =
      'paddleocr_onnx_rec_model.onnx'; //'paddleocr_onnx_rec_model.onnx.enc';
  static const double detThresh = 0.3;
  static const int maxCandidates = 1000;
  static const int minSize = 3;
  static const double boxThresh = 0.2;
  static const double unclipRatio = 1.6;
  static const int recBatchNum = 6;
  static const List<int> recImageShape = [3, 48, 320];
  static const String characterDictPath = "assets/latin_dict.txt";
  static const bool useSpaceChar = true;
  static const double dropScore = 0.4;

  // Utility functions

  // Create a sorted list of indices based on the values in the provided list
  static List<int> argsort(List<double> list) {
    List<int> indices = List<int>.generate(list.length, (i) => i);
    indices.sort((a, b) => list[a].compareTo(list[b]));
    return indices;
  }

  // Sort points by their x-coordinates
  static int compareFunction(List<int> a, List<int> b) {
    return a[0].compareTo(b[0]);
  }

  static void drawLabel(img.Image image, toWrite, x, y) {
    // Label drawing
    img.drawString(
      image,
      toWrite,
      font: img.arial14,
      x: x,
      y: y,
      color: img.ColorRgb8(255, 0, 0),
    );
  }

  static void drawRectangle(img.Image image, Map<String, dynamic> r) {
    // Rectangle drawing
    img.drawRect(
      image,
      x1: r["x1"],
      y1: r["y1"],
      x2: r["x2"],
      y2: r["y2"],
      color: img.ColorRgb8(255, 0, 0),
      thickness: 3,
    );
  }

  // Calculate the euclidean distance
  static double euclideanDistance(List<int> point1, List<int> point2) {
    return math.sqrt(math.pow(point2[0] - point1[0], 2) +
        math.pow(point2[1] - point1[1], 2));
  }

  // Expand a rectangle by a given distance in all directions
  static List<List<int>> expandRectangle(
      List<Tuple2<double, double>> box, double distance) {
    return [
      [
        (box[0].item1 - distance).toInt(),
        (box[0].item2 - distance).toInt()
      ], // top left corner
      [
        (box[1].item1 + distance).toInt(),
        (box[1].item2 - distance).toInt()
      ], // top right corner
      [
        (box[2].item1 + distance).toInt(),
        (box[2].item2 + distance).toInt()
      ], // bottom right corner
      [
        (box[3].item1 - distance).toInt(),
        (box[3].item2 + distance).toInt()
      ] // bottom left corner
    ];
  }

  // Extracts the corner points from a bounding box representation
  static List<Tuple2<double, double>> getBoxPoints(List<dynamic> boundingBox) {
    // Extract center coordinates and dimensions from the bounding box
    Tuple2<double, double> center = boundingBox[0];
    Tuple2<double, double> dimensions = boundingBox[1];
    double width = dimensions.item1;
    double height = dimensions.item2;
    double centerX = center.item1;
    double centerY = center.item2;

    // Calculate corner points of the box
    List<Tuple2<double, double>> points = [
      Tuple2(centerX - width / 2, centerY - height / 2),
      Tuple2(centerX + width / 2, centerY - height / 2),
      Tuple2(centerX - width / 2, centerY + height / 2),
      Tuple2(centerX + width / 2, centerY + height / 2)
    ];

    // Sort points based on x-coordinate for consistency
    points.sort((a, b) => a.item1.compareTo(b.item1));
    return points;
  }

  // Computes the minimal bounding box for a given contour and returns its side length and corner points
  static Map<String, dynamic> getMiniBox(List<List<int>> contour) {
    // Calculate the minimum area rectangle for the contour
    List<dynamic> boundingBox = minAreaRect(contour);

    // Extract the four corner points of the bounding box
    List<Tuple2<double, double>> points = getBoxPoints(boundingBox);

    // Determine the ordering of the points to form a proper rectangle
    // The points are ordered in such a way that they form a continuous loop around the box
    int index1, index2, index3, index4;
    if (points[1].item2 > points[0].item2) {
      index1 = 0;
      index4 = 1;
    } else {
      index1 = 1;
      index4 = 0;
    }
    if (points[3].item2 > points[2].item2) {
      index2 = 2;
      index3 = 3;
    } else {
      index2 = 3;
      index3 = 2;
    }

    // Rearrange the points in the correct order
    List<Tuple2<double, double>> box = [
      points[index1],
      points[index2],
      points[index3],
      points[index4]
    ];

    // Calculate the shorter side of the bounding box
    double sside = math.min(boundingBox[1].item1, boundingBox[1].item2);

    return {'sside': sside, 'box': box};
  }

  // Get file path
  static Future<String> getModelPath(String filename) async {
    final directory = await getApplicationDocumentsDirectory();
    final file = File('${directory.path}/$filename');
    return file.path;
  }

  // Improve Ocr text prediction with custom rules
  static List<dynamic> improveTextPrediction(
      List<Tuple2<String, double>> filterRecRes) {
    List<String> lNumbers = [
      "0",
      "1",
      "2",
      "3",
      "4",
      "5",
      "6",
      "7",
      "8",
      "9",
      '-'
    ];
    List<String> lFibre = ["F", "I", "B", "R", "E"];
    List<String> l31 = ["3", "1", "5", "T", "I", "S"];
    var dNToC = {
      "0": ["O"],
      "1": ["T", "I"],
      "2": ["P"],
      "3": ["S"],
      "4": ["A"],
      "5": ["E", "S"],
      "6": ["G"],
      "8": ["B"]
    };
    var dCToN = {
      "S": ["3", "5"],
      "E": ["5", "8"],
      "L": ["1"],
      "I": ["1"],
      "T": ["1"],
      "O": ["0"],
      "B": ["8"],
      "G": ["6"],
      "Q": ["0"],
      "h": ["1"],
      "z": ["2"],
      "Z": ["2"]
    };
    List<String> txtEtiquette = [];
    List<double> lConfidenceOcr = [];
    for (var t in filterRecRes) {
      double confidence = t.item2;
      if (confidence > 0.5) {
        String txt = t.item1;

        // Only process on text with more than one character (impossible to have only 1 character, must be an error)
        if (txt.length > 1) {
          if (txt.length <= 2) {
            continue;
          }
          List<String> lTxt = txt.split('');

          // Change "=", "*", "~" or ":" to "-"
          lTxt = lTxt
              .map((x) => ["=", "*", "~", ":"].contains(x) ? "-" : x)
              .toList();
          lTxt = lTxt.map((x) => x != '.' ? x : '').toList();

          // If number between 2 str -> change to associated number (same vice versa) (for the first element check i+1 and i+2 and for the last do nothing (impossible to know))
          int maxI = lTxt.length - 1;
          int i = 0;
          while (i < lTxt.length) {
            String c = lTxt[i];
            if (i == 0 && i != maxI && maxI + 1 > 2) {
              // Check the two following elements
              if (lNumbers.contains(lTxt[i + 1]) &&
                  lNumbers.contains(lTxt[i + 2]) &&
                  !lNumbers.contains(lTxt[i])) {
                // c to numb
                c = dCToN[lTxt[i].toUpperCase()]?.first ?? lTxt[i];
                lTxt[i] = c;
              } else if (!lNumbers.contains(lTxt[i + 1]) &&
                  !lNumbers.contains(lTxt[i + 2]) &&
                  lNumbers.contains(lTxt[i])) {
                // c to char
                c = dNToC[lTxt[i]]?.first ?? lTxt[i];
                lTxt[i] = c;
              }
            } else if (i == maxI || maxI + 1 <= 2) {
              // Nothing to do
              lTxt[i] = c;
            } else {
              // Check the previous and following elements
              if (lNumbers.contains(lTxt[i - 1]) &&
                  lNumbers.contains(lTxt[i + 1]) &&
                  !lNumbers.contains(lTxt[i])) {
                // c to numb
                c = dCToN[lTxt[i].toUpperCase()]?.first ?? lTxt[i];
                lTxt[i] = c;
              } else if (!lNumbers.contains(lTxt[i - 1]) &&
                  !lNumbers.contains(lTxt[i + 1]) &&
                  lNumbers.contains(lTxt[i])) {
                // c to char
                c = dNToC[lTxt[i]]?.first ?? lTxt[i];
                lTxt[i] = c;
              }
            }

            // Delete single characters
            int cAlone = 0;
            // If l is the first element of the list
            cAlone += (i - 1) == -1 ? 1 : 0;
            // If l is the last element of the list
            cAlone +=
                (i + 1) > (lTxt.length - 1) ? 1 : (lTxt[i + 1] == " " ? 1 : 0);
            if (cAlone > 1) {
              lTxt[i] = "";
            }
            i++;
          }

          // If 3 similar elements with lFibre and 1 with l31 -> change to FIBRE31
          var lSameFibre = lTxt.toSet().intersection(lFibre.toSet()).toList();
          if (lSameFibre.length >= 3 && maxI + 1 <= 8) {
            var lTxtUpdated =
                lTxt.toSet().difference(lSameFibre.toSet()).toList();
            if (lTxtUpdated.toSet().intersection(l31.toSet()).isNotEmpty) {
              lTxt = "FIBRE31".split('');
            }
          }

          // If 7 numbers and third elements = 1 or 7 or, then change to "/" (certainly a date)
          if (lTxt.length == 7) {
            if (lTxt.every((element) => lNumbers.contains(element))) {
              if (["1", "7", ","].contains(lTxt[2])) {
                lTxt[2] = "/";
              }
            }
          }

          if (lTxt.length >= 3) {
            // If in the 3 first elements 2 looks like "PBO"
            int iPbo = 0;
            iPbo += ["P"].contains(lTxt[0]) ? 1 : 0;
            iPbo += ["B", "8"].contains(lTxt[1]) ? 1 : 0;
            iPbo += ["O", "0"].contains(lTxt[2]) ? 1 : 0;
            if (iPbo >= 2) {
              var firstThreeChars = lTxt.sublist(0, 3).toSet();
              if (firstThreeChars
                  .intersection({"P", "B", "O"}.toSet())
                  .isNotEmpty) {
                lTxt.replaceRange(0, 3, ["P", "B", "O"]);
              }
            }

            // If in the 3 first elements 2 looks like "CDI"
            int iCdi = 0;
            iCdi += ["C"].contains(lTxt[0]) ? 1 : 0;
            iCdi += ["D"].contains(lTxt[1]) ? 1 : 0;
            iCdi += ["I", "T", "1", "l", "E", "K", "L", "F"].contains(lTxt[2])
                ? 1
                : 0;
            if (iCdi >= 2) {
              var firstThreeChars = lTxt.sublist(0, 3).toSet();
              if (firstThreeChars
                  .intersection({"C", "D", "I"}.toSet())
                  .isNotEmpty) {
                if (!["E", "K", "L", "F"].contains(lTxt[2])) {
                  lTxt.replaceRange(0, 3, ["C", "D", "I"]);
                } else {
                  lTxt.replaceRange(0, 3, ["C", "D", "I-"]);
                }
              }
            }
          }

          // If text looks like "OCTO"
          if (lTxt.length >= 4) {
            int iOcto = 0;
            iOcto += ["Q", "0", "O"].contains(lTxt[0]) ? 1 : 0;
            iOcto += ["C"].contains(lTxt[1]) ? 1 : 0;
            iOcto += ["T", "1", "I"].contains(lTxt[2]) ? 1 : 0;
            iOcto += ["Q", "0", "O"].contains(lTxt[3]) ? 1 : 0;
            if (iOcto >= 3) {
              var firstFourChars = lTxt.sublist(0, 4).toSet();
              if (firstFourChars
                      .intersection({"O", "C", "T", "O"}.toSet())
                      .length >=
                  2) {
                lTxt.replaceRange(0, 4, ["O", "C", "T", "O"]);
              }
            }
          }

          // Create txt from list
          txt = lTxt.join('');
          // Delete useless spaces
          if (txt.isNotEmpty) {
            if (txt.startsWith(' ')) {
              txt = txt.substring(1);
            }
            if (txt.endsWith(' ')) {
              txt = txt.substring(0, txt.length - 1);
            }

            txt = txt.replaceAll('--', '-');
            txtEtiquette.add(txt);
            lConfidenceOcr.add(double.parse(confidence.toStringAsFixed(3)));
          }
        }
      }
    }

    return [txtEtiquette, lConfidenceOcr];
  }

  // Calculates the smallest bounding rectangle for a contour
  static List<dynamic> minAreaRect(List<List<int>> contour) {
    int minX = contour[0][0];
    int maxX = contour[0][0];
    int minY = contour[0][1];
    int maxY = contour[0][1];

    // Iterate through each point in the contour to find the extreme coordinates
    for (var point in contour) {
      if (point[0] < minX) {
        minX = point[0];
      }
      if (point[0] > maxX) {
        maxX = point[0];
      }
      if (point[1] < minY) {
        minY = point[1];
      }
      if (point[1] > maxY) {
        maxY = point[1];
      }
    }
    double width = maxX - minX + 1;
    double height = maxY - minY + 1;
    double centerX = minX + width / 2.0;
    double centerY = minY + height / 2.0;

    // Return the center, dimensions, and angle (0.0 since it's axis-aligned)
    return [Tuple2(centerX, centerY), Tuple2(width, height), 0.0];
  }

  // Calculate the area of a polygon given its vertices
  static double polygonArea(List<Tuple2<double, double>> points) {
    int n = points.length;
    double area = 0.0;
    for (int i = 0; i < n; i++) {
      int next = (i + 1) % n;
      // Apply the shoelace formula for area calculation
      area += points[i].item1 * points[next].item2;
      area -= points[next].item1 * points[i].item2;
    }
    return area.abs() / 2.0;
  }

  // Calculate the perimeter of a polygon given its vertices
  static double polygonPerimeter(List<Tuple2<double, double>> points) {
    int n = points.length;
    double perimeter = 0.0;
    for (int i = 0; i < n; i++) {
      int next = (i + 1) % n;
      // Calculate the distance between the current point and the next point
      double dx = points[next].item1 - points[i].item1;
      double dy = points[next].item2 - points[i].item2;
      perimeter += math.sqrt(dx * dx + dy * dy);
    }
    return perimeter;
  }

  // Scale and clip the coordinates of a box
  static List<List<int>> scaleAndClipBox(
      List<List<double>> box, int width, int srcH, int srcW, int height) {
    for (var point in box) {
      // Scale the x-coordinate and clamp it within the range [0, srcW]
      point[0] = (point[0] / width * srcW).round().clamp(0, srcW).toDouble();
      // Scale the y-coordinate and clamp it within the range [0, srcH]
      point[1] = (point[1] / height * srcH).round().clamp(0, srcH).toDouble();
    }
    return box.map((e) => [e[0].toInt(), e[1].toInt()]).toList();
  }
}
