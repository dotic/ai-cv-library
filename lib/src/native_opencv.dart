import 'dart:ffi' as ffi;
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'dart:typed_data';

import 'package:tuple/tuple.dart';

// C function signatures
typedef _CGetPerspectiveTransformFunc = ffi.Void Function(
  ffi.Pointer<ffi.Double>,
  ffi.Pointer<ffi.Double>,
  ffi.Pointer<ffi.Double>,
);

// Dart function signatures
typedef GetPerspectiveTransformFunc = void Function(
    ffi.Pointer<ffi.Double>, ffi.Pointer<ffi.Double>, ffi.Pointer<ffi.Double>);

// Loading opencv
ffi.DynamicLibrary _lib = Platform.isAndroid
    ? ffi.DynamicLibrary.open('libnative_opencv.so')
    : ffi.DynamicLibrary.process();

// Looking for the functions
final GetPerspectiveTransformFunc getPerspectiveTransform = _lib
    .lookup<ffi.NativeFunction<_CGetPerspectiveTransformFunc>>('get_perspective_transform')
    .asFunction();

// A helper function that converts a list of points to a Pointer<Double>
ffi.Pointer<ffi.Double> _convertPointsToNative(List<List<int>> points) {
  final pointer = calloc<ffi.Double>(points.length * 2);
  for (int i = 0; i < points.length; i++) {
    pointer[i * 2] = points[i][0].toDouble();
    pointer[i * 2 + 1] = points[i][1].toDouble();
  }
  return pointer;
}

// Wrapper function to call getPerspectiveTransform
List<List<double>> callGetPerspectiveTransform(
    List<List<int>> srcPoints, List<List<int>> dstPoints) {
  final srcPointer = _convertPointsToNative(srcPoints);
  final dstPointer = _convertPointsToNative(dstPoints);
  final matrixPointer = calloc<ffi.Double>(9); // Perspective transform is a 3x3 matrix

  getPerspectiveTransform(srcPointer, dstPointer, matrixPointer);

  // Convert the resulting matrix back to a Dart list of lists (matrix)
  List<List<double>> matrix =
      List.generate(3, (i) => List.generate(3, (j) => matrixPointer[i * 3 + j]));

  // Free the native memory
  calloc.free(srcPointer);
  calloc.free(dstPointer);
  calloc.free(matrixPointer);

  return matrix;
}

typedef _CWarpPerspectiveAndGetBufferFunc = ffi.Void Function(
    ffi.Pointer<ffi.Uint8>, // Image data
    ffi.Int32, // Image width
    ffi.Int32, // Image height
    ffi.Pointer<ffi.Double>, // Perspective matrix
    ffi.Int32, // Output width
    ffi.Int32, // Output height
    ffi.Pointer<ffi.Uint8> // Output image data buffer
    );

// Dart function signature
typedef _WarpPerspectiveAndGetBufferFunc = void Function(
    ffi.Pointer<ffi.Uint8>, int, int, ffi.Pointer<ffi.Double>, int, int, ffi.Pointer<ffi.Uint8>);

// Looking for the function in the library
final _WarpPerspectiveAndGetBufferFunc _warpPerspectiveAndGetBuffer = _lib
    .lookup<ffi.NativeFunction<_CWarpPerspectiveAndGetBufferFunc>>('warpPerspectiveAndGetBuffer')
    .asFunction();

ffi.Pointer<ffi.Uint8> _convertImageToNative(Uint8List imageData) {
  final ptr = calloc<ffi.Uint8>(imageData.length);
  final nativeUint8List = ptr.asTypedList(imageData.length);
  nativeUint8List.setAll(0, imageData);
  return ptr;
}

List<double> flattenMatrix(List<List<double>> matrix) {
  return matrix.expand((i) => i).toList();
}

List<List<List<double>>> warpPerspective(Uint8List srcImageData, int srcWidth, int srcHeight,
    List<List<double>> transformMatrix, int outWidth, int outHeight) {
  // Flatten the transformation matrix
  final flatMatrix = flattenMatrix(transformMatrix);

  // Convert source image data into a native pointer
  final srcImagePtr = _convertImageToNative(srcImageData);

  // Convert the transformation matrix into a native pointer
  final matrixPtr = calloc<ffi.Double>(flatMatrix.length);
  for (int i = 0; i < flatMatrix.length; i++) {
    matrixPtr[i] = flatMatrix[i];
  }

  // Preparing the buffer to receive the transformed image
  final outBufferPtr = calloc<ffi.Uint8>(outWidth * outHeight * 3);

  // Call the C++ function to perform the perspective transformation and retrieve the transformed image
  _warpPerspectiveAndGetBuffer(
      srcImagePtr, srcWidth, srcHeight, matrixPtr, outWidth, outHeight, outBufferPtr);

  // Convert result to Uint8List
  final resultUint8List = outBufferPtr.asTypedList(outWidth * outHeight * 3);

  // Build 3D list from transformed image buffer
  final List<List<List<double>>> image3D = List.generate(
    outHeight,
    (_) => List.generate(
      outWidth,
      (_) => List.generate(3, (_) => 0.0),
    ),
  );

  for (int y = 0; y < outHeight; y++) {
    for (int x = 0; x < outWidth; x++) {
      int idx = (y * outWidth + x) * 3;
      image3D[y][x][0] = resultUint8List[idx].toDouble();
      image3D[y][x][1] = resultUint8List[idx + 1].toDouble();
      image3D[y][x][2] = resultUint8List[idx + 2].toDouble();
    }
  }

  // Free allocated memory
  calloc.free(srcImagePtr);
  calloc.free(matrixPtr);
  calloc.free(outBufferPtr);

  return image3D;
}

// C function signature for finding contours in an image
typedef _CFindContoursFunc = ffi.Int32 Function(
  ffi.Pointer<ffi.Uint8>, // Image data
  ffi.Int32, // Image width
  ffi.Int32, // Image height
  ffi.Pointer<ffi.Int32>, // Output buffer for contour data
  ffi.Int32, // Output buffer size
);

// Dart function signature for finding contours
typedef _FindContoursFunc = int Function(
  ffi.Pointer<ffi.Uint8>,
  int,
  int,
  ffi.Pointer<ffi.Int32>,
  int,
);

// Function lookup for findContours in the native library
final _FindContoursFunc _findContours =
    _lib.lookup<ffi.NativeFunction<_CFindContoursFunc>>('findContours').asFunction();

// Function to find contours in an image mask
List<List<List<int>>> findContoursInMask(Uint8List srcImageData, int srcWidth, int srcHeight) {
  final srcImagePtr = _convertImageToNative(srcImageData);

  // Estimate the size required for the output buffer
  final int estimatedSize = srcWidth * srcHeight;
  final outBufferPtr = calloc<ffi.Int32>(estimatedSize);

  // Call C++ function
  int written = _findContours(srcImagePtr, srcWidth, srcHeight, outBufferPtr, estimatedSize);

  // Convert results into a list of Dart points
  var result = List<List<int>>.empty(growable: true);
  for (int i = 0; i < written; i += 2) {
    result.add([outBufferPtr[i], outBufferPtr[i + 1]]);
  }

  // Contour deserialization
  var contours = <List<List<int>>>[];
  var currentContour = <List<int>>[];

  for (int i = 0; i < written; i += 2) {
    int x = outBufferPtr[i];
    int y = outBufferPtr[i + 1];

    if (x == -1 && y == -1) {
      // End of a contour
      contours.add(currentContour);
      currentContour = <List<int>>[];
    } else {
      currentContour.add([x, y]);
    }
  }

  // Adding the last contour if it's not empty
  if (currentContour.isNotEmpty) {
    contours.add(currentContour);
  }

  // Free the allocated native memory
  calloc.free(srcImagePtr);
  calloc.free(outBufferPtr);

  return contours;
}

// C function signature for finding the minimum area rectangle
typedef _CMinAreaRectFunc = ffi.Void Function(
  ffi.Pointer<ffi.Int32>, // Points
  ffi.Int32, // Number of points
  ffi.Pointer<ffi.Double>, // Output buffer
);

// Dart function signature for minAreaRect
typedef _MinAreaRectFunc = void Function(
  ffi.Pointer<ffi.Int32>,
  int,
  ffi.Pointer<ffi.Double>,
);

// Function lookup for minAreaRect in the native library
final _MinAreaRectFunc _minAreaRect =
    _lib.lookup<ffi.NativeFunction<_CMinAreaRectFunc>>('minAreaRect').asFunction();

// Function to find the minimum area rectangle for a given contour
List<dynamic> minAreaRect(List<List<int>> contour) {
  // Flatten the contour points for processing
  final pointsFlat = contour.expand((p) => p).toList();
  final pointsPtr = calloc<ffi.Int32>(pointsFlat.length);
  final outBufferPtr = calloc<ffi.Double>(5); // cx, cy, width, height, angle

  // Copy contour points into the native pointer
  for (int i = 0; i < pointsFlat.length; i++) {
    pointsPtr[i] = pointsFlat[i];
  }

  // Call the native function to compute the minimum area rectangle
  _minAreaRect(pointsPtr, contour.length, outBufferPtr);

  // Extract the rectangle properties from the output buffer
  Tuple2<double, double> center = Tuple2(outBufferPtr[0], outBufferPtr[1]);
  Tuple2<double, double> size = Tuple2(outBufferPtr[2], outBufferPtr[3]);
  double angle = outBufferPtr[4];

  // Free the allocated native memory
  calloc.free(pointsPtr);
  calloc.free(outBufferPtr);

  return [center, size, angle];
}

// C function signature for filling a polygon in a mask
typedef _CFillPolyFunc = ffi.Void Function(
  ffi.Pointer<ffi.Uint8>, // Mask data
  ffi.Int32, // Mask width
  ffi.Int32, // Mask height
  ffi.Pointer<ffi.Int32>, // Box points
  ffi.Int32, // Number of points in the box
);

// Dart function signature for fillPoly
typedef _FillPolyFunc = void Function(
  ffi.Pointer<ffi.Uint8>,
  int,
  int,
  ffi.Pointer<ffi.Int32>,
  int,
);

// Function lookup for fillPoly in the native library
final _FillPolyFunc _fillPoly =
    _lib.lookup<ffi.NativeFunction<_CFillPolyFunc>>('fillPoly').asFunction();

// Function to fill a polygon in a mask
List<List<int>> fillPoly(List<List<int>> mask, List<List<double>> box) {
  // Flatten the mask and box data for native processing
  final maskFlat = mask.expand((row) => row).toList();
  final boxFlat = box.expand((point) => point).map((p) => p.toInt()).toList();

  final maskPtr = calloc<ffi.Uint8>(maskFlat.length);
  final boxPtr = calloc<ffi.Int32>(boxFlat.length);

  // Initialize mask buffer with existing data
  for (int i = 0; i < maskFlat.length; i++) {
    maskPtr[i] = maskFlat[i];
  }

  // Prepare and pass the box buffer
  for (int i = 0; i < boxFlat.length; i++) {
    boxPtr[i] = boxFlat[i];
  }

  // Call the native function to fill the polygon in the mask
  _fillPoly(maskPtr, mask[0].length, mask.length, boxPtr, boxFlat.length);

  // Rebuild the modified mask from the buffer
  List<List<int>> modifiedMask = [];
  for (int i = 0; i < mask.length; i++) {
    List<int> row = [];
    for (int j = 0; j < mask[0].length; j++) {
      row.add(maskPtr[i * mask[0].length + j]);
    }
    modifiedMask.add(row);
  }

  // Free the allocated native memory
  calloc.free(maskPtr);
  calloc.free(boxPtr);

  return modifiedMask;
}

// C function signature for image ocr preprocessing
typedef _CProcessImageFunc = ffi.Void Function(
    ffi.Pointer<ffi.Uint8>, // Image data
    ffi.Int32, // Image width
    ffi.Int32, // Image height
    ffi.Pointer<ffi.Uint8> // Output buffer
    );

// Dart function signature
typedef _ProcessImageFunc = void Function(ffi.Pointer<ffi.Uint8>, int, int, ffi.Pointer<ffi.Uint8>);

// Looking up the native 'preProcessImage' function from the library
final _ProcessImageFunc _processImage =
    _lib.lookup<ffi.NativeFunction<_CProcessImageFunc>>('preProcessImage').asFunction();

Uint8List preProcessImage(Uint8List imageData, int width, int height) {
  // Allocating native memory
  final imageDataPtr = calloc<ffi.Uint8>(imageData.length);
  final outputPtr = calloc<ffi.Uint8>(width * height);

  final nativeUint8List = imageDataPtr.asTypedList(imageData.length);
  nativeUint8List.setAll(0, imageData);

  // Calling the native function to process the image
  _processImage(imageDataPtr, width, height, outputPtr);

  final resultUint8List = outputPtr.asTypedList(width * height);
  Uint8List result = Uint8List.fromList(resultUint8List);

  // Free the allocated native memory
  calloc.free(imageDataPtr);
  calloc.free(outputPtr);

  return result;
}
