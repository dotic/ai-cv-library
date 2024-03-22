import 'native_opencv_platform_interface.dart';

class NativeOpencv {
  Future<String?> getPlatformVersion() {
    return NativeOpencvPlatform.instance.getPlatformVersion();
  }
}
