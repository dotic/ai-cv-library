import 'dart:developer';
import 'dart:io';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tuple/tuple.dart';
import 'native_opencv.dart' as cv2;
import 'dart:math' as math;
import 'package:image_picker/image_picker.dart';

class ImageProcessing {
  final imagePicker = ImagePicker();

  static img.Image loadImage(String imagePath) {
    log('Analysing image...');
    // Reading image bytes from file
    final imageData = File(imagePath).readAsBytesSync();
    // Decoding image
    final image = img.decodeImage(imageData);

    //Resizing and add border [640, 640]
    final img.Image imageInput = ImageProcessing.letterbox(image!);

    return imageInput;
  }

  // Resize and add borders to an image
  static img.Image letterbox(img.Image image,
      {int outWidth = 640,
      int outHeight = 640,
      List<int> borderColor = const [114, 114, 114],
      bool scaleUp = true}) {
    // Calculate the scaling ratio to maintain proportions
    double ratio = calculateAspectRatioFit(image.width, image.height, outWidth, outHeight, scaleUp);

    // Calculate new dimensions
    int newWidth = (image.width * ratio).round();
    int newHeight = (image.height * ratio).round();

    // Resize image
    img.Image resizedImage = img.copyResize(image,
        width: newWidth, height: newHeight, interpolation: img.Interpolation.linear);

    // Create a new image with the desired background color
    img.Image borderedImage = img.Image(width: outWidth, height: outHeight);
    img.Color colorToFill = img.ColorRgba8(borderColor[0], borderColor[1], borderColor[2], 255);
    img.fill(borderedImage, color: colorToFill);

    // Calculate positioning for drawing
    int padLeft = (outWidth - newWidth) ~/ 2;
    int padTop = (outHeight - newHeight) ~/ 2;

    // Draw resized image on new image with background color
    for (int y = 0; y < newHeight; y++) {
      for (int x = 0; x < newWidth; x++) {
        borderedImage.setPixel(padLeft + x, padTop + y, resizedImage.getPixel(x, y));
      }
    }
    return borderedImage;
  }

  static double calculateAspectRatioFit(
      int srcWidth, int srcHeight, int maxWidth, int maxHeight, bool scaleUp) {
    double widthRatio = maxWidth / srcWidth;
    double heightRatio = maxHeight / srcHeight;
    double ratio = widthRatio < heightRatio ? widthRatio : heightRatio;

    if (!scaleUp && ratio > 1) {
      return 1.0; // Do not enlarge if scaleUp is false
    }
    return ratio;
  }

  // Convert an image representation from channels-last (HWC) format to channels-first (CHW) format
  static List<List<List<num>>> convertChannelsLastToChannelsFirst(
      List<List<List<num>>> channelsLast) {
    final numRows = channelsLast.length;
    final numCols = channelsLast[0].length;
    final numChannels = channelsLast[0][0].length;

    final channelsFirst = List.generate(numChannels, (c) {
      return List.generate(numRows, (r) {
        return List.generate(numCols, (col) {
          return channelsLast[r][col][c];
        });
      });
    });

    return channelsFirst;
  }

  // Yolov7 requires input normalized between 0 and 1
  static List<List<List<num>>> convertImageToMatrix(img.Image image) {
    return List.generate(
      image.height,
      (y) => List.generate(
        image.width,
        (x) {
          final pixel = image.getPixel(x, y);
          return [pixel.rNormalized, pixel.gNormalized, pixel.bNormalized];
        },
      ),
    );
  }

  // Preprocessing for ocr
  static Tuple3<List<List<List<int>>>, List<List<List<double>>>, List<double>>
      preprocessImageForOcr(img.Image imageInput, Map<String, dynamic> r) {
    // Crop etiquette
    final croppedImage = img.copyCrop(imageInput,
        x: r["x1"]!, y: r["y1"]!, width: r["x2"]! - r["x1"]!, height: r["y2"]! - r["y1"]!);

    // Convert to matrix and preprocess
    List<List<List<int>>> imageInputList = convertImageToMatrix(croppedImage)
        .map((list2D) =>
            list2D.map((list1D) => list1D.map((value) => (value * 255).toInt()).toList()).toList())
        .toList();

    // Apply preprocess CLAHE, detail enhance, gray scale, bilateral filter
    Uint8List imageData = convertImageToUint8List(imageInputList);
    Uint8List processedImageData =
        cv2.preProcessImage(imageData, imageInputList[0].length, imageInputList.length);
    List<List<List<double>>> dataImageUp = convertUint8ListTo3DList(
        processedImageData, imageInputList[0].length, imageInputList.length);

    // Convert to 3-channel grayscale image
    List<List<List<int>>> rgbImage = convertTo3Channels(dataImageUp);
    imageInputList = rgbImage;

    // Resize logic and normalize
    int srcH = imageInputList.length;
    int srcW = imageInputList[0].length;
    int limitSideLen = 960;
    int h = imageInputList.length;
    int w = imageInputList[0].length;
    double ratio = (math.max(w, h) > limitSideLen)
        ? (math.max(w, h) == h ? limitSideLen / h : limitSideLen / w)
        : 1.0;
    int resizeH = math.max((h * ratio).toInt() ~/ 32 * 32, 32);
    int resizeW = math.max((w * ratio).toInt() ~/ 32 * 32, 32);
    final imageInputResized = img.copyResize(croppedImage, width: resizeW, height: resizeH);

    List<List<List<double>>> imageInputResizedList = convertImageToMatrix(imageInputResized)
        .map((list2D) => list2D
            .map((list1D) => list1D.map((value) => (value * 255).toDouble()).toList())
            .toList())
        .toList();

    // Normalize image
    double ratioH = resizeH / h.toDouble();
    double ratioW = resizeW / w.toDouble();
    List<double> dataShape = [srcH.toDouble(), srcW.toDouble(), ratioH, ratioW];
    double scale = 1 / 255;
    List<List<List<double>>> mean = [
      [
        [0.485, 0.456, 0.406]
      ]
    ];
    List<List<List<double>>> std = [
      [
        [0.229, 0.224, 0.225]
      ]
    ];
    List<List<List<double>>> dataImage = processImage(imageInputResizedList, scale, mean, std);

    // Convert channels
    List<List<List<double>>> dataImageCHW = convertChannelsLastToChannelsFirst(dataImage)
        .map(
            (list2D) => list2D.map((list) => list.map((item) => item.toDouble()).toList()).toList())
        .toList();

    return Tuple3(imageInputList, dataImageCHW, dataShape);
  }

  // Convert a 3D list of pixel values into a flat Uint8List
  static Uint8List convertImageToUint8List(List<List<List<int>>> image) {
    int height = image.length;
    int width = image[0].length;
    int channels = image[0][0].length;
    Uint8List imageBuffer = Uint8List(width * height * channels);
    int bufferIndex = 0;

    // Iterate over each pixel and each channel to flatten the image data into the buffer
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        for (int c = 0; c < channels; c++) {
          int pixelValue = image[y][x][c].clamp(0, 255);
          imageBuffer[bufferIndex++] = pixelValue;
        }
      }
    }
    return imageBuffer;
  }

  // Convert a Uint8List back into a 3D list structure representing an image
  static List<List<List<double>>> convertUint8ListTo3DList(
      Uint8List imageData, int width, int height) {
    List<List<List<double>>> image =
        List.generate(height, (_) => List.generate(width, (_) => List.generate(1, (_) => 0.0)));

    // Iterate over the Uint8List and populate the 3D list
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int pixelValue = imageData[y * width + x];
        image[y][x][0] = pixelValue.toDouble(); // / 255.0;
      }
    }
    return image;
  }

  // Converts a single-channel grayscale image into a three-channel RGB image
  static List<List<List<int>>> convertTo3Channels(List<List<List<double>>> grayImage) {
    int height = grayImage.length;
    int width = grayImage[0].length;
    List<List<List<int>>> threeChannelImage = List.generate(
        height, (_) => List.generate(width, (_) => List.generate(3, (_) => 0))); // 3 -> RGB
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int pixelValue = grayImage[y][x][0].toInt(); // Pixel value in grayscale
        for (int c = 0; c < 3; c++) {
          threeChannelImage[y][x][c] = pixelValue; // Fill all three channels with the same value
        }
      }
    }
    return threeChannelImage;
  }

  // Normalizes an image by scaling, subtracting mean, and dividing by standard deviation
  static List<List<List<double>>> processImage(List<List<List<double>>> img, double scale,
      List<List<List<double>>> mean, List<List<List<double>>> std) {
    int height = img.length;
    int width = img[0].length;
    int channels = img[0][0].length;

    // Initialize a 3D list to store the normalized data
    List<List<List<double>>> dataImage = List.generate(
        height, (_) => List.generate(width, (_) => List.generate(channels, (_) => 0.0)));

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        for (int c = 0; c < channels; c++) {
          // Apply normalization: scale the value, subtract the mean, and divide by the standard deviation
          dataImage[y][x][c] = ((img[y][x][c] * scale) - mean[0][0][c]) / std[0][0][c];
        }
      }
    }
    return dataImage;
  }

  // Converts a 2D boolean mask to a Uint8List
  static Uint8List convertBoolMaskToUint8List(List<List<bool>> mask) {
    int height = mask.length;
    int width = mask[0].length;
    // Create a buffer for image data and define a typed view to facilitate data writing.
    Uint8List imageBuffer = Uint8List(width * height);
    var bufferIndex = 0;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        // Convert each Boolean to 255 (white) or 0 (black).
        imageBuffer[bufferIndex++] = mask[y][x] ? 255 : 0;
      }
    }
    return imageBuffer;
  }

  // Converts a 3D nested list (representing an image) to Uint8List format
  static Uint8List convertNestedListToUint8List(List<List<List<int>>> image) {
    int height = image.length;
    int width = image[0].length;
    int numChannels = image[0][0].length;
    // Create a buffer for image data and define a typed view to facilitate data writing.
    Uint8List imageBuffer = Uint8List(width * height * numChannels);
    var bufferIndex = 0;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        for (int c = 0; c < numChannels; c++) {
          imageBuffer[bufferIndex++] = image[y][x][c].clamp(0, 255);
        }
      }
    }
    return imageBuffer;
  }

  // Rotates a 3D list representing an image by 90 degrees counter-clockwise
  static List<List<List<double>>> rotate90DegreesLeft(List<List<List<double>>> image) {
    int height = image.length;
    int width = image[0].length;

    // Create a new image with swapped width and height dimensions
    var rotatedImage =
        List.generate(width, (_) => List.generate(height, (_) => List.generate(3, (_) => 0.0)));

    // Perform the rotation
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        rotatedImage[x][height - 1 - y] = image[y][x];
      }
    }
    return rotatedImage;
  }

  // Converts a 3D list representing an image crop to an image object
  static img.Image imgCropListToImageObject(List<List<List<num>>> imgCropList) {
    int h = imgCropList.length;
    int w = imgCropList[0].length;
    img.Image image = img.Image(width: w, height: h);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        int r = (imgCropList[y][x][0]).toInt();
        int g = (imgCropList[y][x][1]).toInt();
        int b = (imgCropList[y][x][2]).toInt();
        image.setPixelRgba(x, y, r, g, b, 255);
      }
    }
    return image;
  }

  // Converts an image object to a 3D list representing the image crop
  static List<List<List<double>>> imageObjectToImgCropList(img.Image image) {
    int h = image.height;
    int w = image.width;
    List<List<List<double>>> imgCropList = List.generate(
        h,
        (y) => List.generate(w, (x) {
              img.Pixel pixelColor = image.getPixel(x, y);
              return [
                (pixelColor.r).toDouble(),
                (pixelColor.g).toDouble(),
                (pixelColor.b).toDouble()
              ];
            }));
    return imgCropList;
  }
}
