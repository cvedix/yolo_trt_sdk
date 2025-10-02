#include <yolov8.hpp>
#include <iostream>

int main() {
    // Basic sanity: default config values
    YoloV8Config cfg;
    if (cfg.topK != 100) {
        std::cerr << "topK default mismatch" << std::endl;
        return 1;
    }
    if (cfg.classNames.size() < 1) {
        std::cerr << "classNames should not be empty" << std::endl;
        return 1;
    }
    // File existence helper should return false for a bogus file
    if (doesFileExist("__this_file_should_not_exist__")) {
        std::cerr << "doesFileExist returned true unexpectedly" << std::endl;
        return 1;
    }
    std::cout << "SDK header test passed" << std::endl;
    return 0;
}

