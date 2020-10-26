#ifndef PINKIE_UTILS_CSRC_HEADER_H
#define PINKIE_UTILS_CSRC_HEADER_H

#ifdef _WIN32
#  define PINKIE_API __declspec(dllexport)
#else
#  define PINKIE_API
#endif


#ifdef _WIN32
#  define PRAGMA __pragma
#else
#  define PRAGMA _Pragma
#endif

#endif //PINKIE_UTILS_CSRC_HEADER_H