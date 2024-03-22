#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint native_opencv.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'native_opencv'
  s.version          = '0.0.1'
  s.summary          = 'A new Flutter plugin project.'
  s.description      = <<-DESC
A new Flutter plugin project opencv.
                       DESC
  s.homepage         = 'http://example.com'
  s.license          = { :type => 'MIT', :file => '../LICENSE' }
  s.author           = { 'Your Company' => 'email@example.com' }
  s.source           = { :path => '.' }
  # s.source_files = 'Classes/**/*'
  s.dependency 'Flutter'
  s.platform = :ios, '12.0'

  # Flutter.framework does not contain a i386 slice.
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++11',
    'CLANG_CXX_LIBRARY' => 'libc++'
  }

  s.user_target_xcconfig = {
     'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
     #'CLANG_CXX_LANGUAGE_STANDARD' => 'c++11',
     #'CLANG_CXX_LIBRARY' => 'libc++',
  }

  # telling linker to include opencv2 framework
  s.xcconfig = {
    'OTHER_LDFLAGS' => '-framework opencv2',
    #'CLANG_CXX_LANGUAGE_STANDARD' => 'c++11',
    #'CLANG_CXX_LIBRARY' => 'libc++'
  }

  s.swift_version = '5.0'

  # telling CocoaPods not to remove framework
  s.preserve_paths = 'opencv2.framework'

  # including OpenCV framework
  s.vendored_frameworks = 'opencv2.framework'

  # including native framework
  s.frameworks = 'AVFoundation'
  #s.frameworks = 'Accelerate', 'AssetsLibrary', 'AVFoundation', 'CoreGraphics', 'CoreImage', 'CoreMedia', 'CoreVideo', 'Foundation', 'QuartzCore', 'UIKit'

  # including C++ library
  s.library = 'c++'

  s.source_files = 'Classes/**/*' #.cpp', 'opencv2.framework/Versions/A/Headers/**/*{.h,.hpp}'
  # s.header_dir = 'opencv2'
  # s.header_mappings_dir = 'opencv2.framework/Versions/A/Headers/'
  # s.requires_arc = false

end
