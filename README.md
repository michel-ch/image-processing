[![Build Status](https://perretb.visualstudio.com/AzurePipelines/_apis/build/status%2FPerretB.ImageProcessingLab?branchName=main)](https://perretb.visualstudio.com/AzurePipelines/_build/latest?definitionId=1&branchName=main)

# Image Processing Lab

Lab exercises for the  [course on image processing](https://perso.esiee.fr/~perretb/I5FM/TAI/) (in French).



## Organization

There are 5 files completed, each one corresponds to a course chapter:

1. ``src/tpHistogram.cpp``: histogram manipulation functions [related course chapter](https://perso.esiee.fr/~perretb/I5FM/TAI/histogramme/index.html) Alternative Link on Local [Traitement d’histogramme](https://github.com/michel-ch/image-processing/blob/main/site/Traitement%20d%E2%80%99histogramme%20%E2%80%94%20Documentation%20Traitement%20et%20analyse%20d'images%201.html)
2. ``src/tpConnectedComponents.cpp`` pixel adjacency and connected components [related course chapter](https://perso.esiee.fr/~perretb/I5FM/TAI/connexity/index.html) Alternative Link on Local [Opérateurs connexes](https://github.com/michel-ch/image-processing/blob/main/site/Op%C3%A9rateurs%20connexes%20%E2%80%94%20Documentation%20Traitement%20et%20analyse%20d'images%201.html)
3. ``src/tpGeometry.cpp`` geometric image transforms [related course chapter](https://perso.esiee.fr/~perretb/I5FM/TAI/geometry/index.html) Alternative Link on Local [Transformations géométriques](https://github.com/michel-ch/image-processing/blob/main/site/Transformations%20g%C3%A9om%C3%A9triques%20%E2%80%94%20Documentation%20Traitement%20et%20analyse%20d'images%201.html)
4. ``src/tpConvolution.cpp`` linear image filters [related course chapter](https://perso.esiee.fr/~perretb/I5FM/TAI/convolution/index.html) Alternative Link on Local [Convolution](https://github.com/michel-ch/image-processing/blob/main/site/Convolution%20%E2%80%94%20Documentation%20Traitement%20et%20analyse%20d'images%201.html)
5. ``src/tpMorphology.cpp`` non-linear image filters [related course chapter](https://perso.esiee.fr/~perretb/I5FM/TAI/morpho/index.html) Alternative Link on Local [Morphologie Mathématique](https://github.com/michel-ch/image-processing/blob/main/site/Morphologie%20Math%C3%A9matique%20%E2%80%94%20Documentation%20Traitement%20et%20analyse%20d'images%201.html)

## How to


### Environment

This lab requires a working *nix environment with a c++ 11 compiler and valgrind.

The library *OpenCV4* is required to compile this project. To install a local version of this library just execute the script ``source ./init.sh``. This script will automatically download the library and set up the environment variable required by the makefile.  

**Important:** If you install **OpenCV** with the script ``source ./init.sh``, you have to run the script in every terminal used to work on this project!

### Compilation

Just run ``make`` in the top directory, this will create several executables in the ``bin`` directory. 

**Tip** When using *make*, use the argument ``-j n`` to launch a parallel compilation on ``n`` cores. For example: ``make -j 4`` while compile up to 4 files in parallel. 

### Executables

Each exercise is associated to its own command line tool. For exemple, the first function to code, ``inverse`` in the file ``src/tpHistogram.cpp``, is associated to the command line tool ``bin/inverse`` generated by *make*.

The usage of the command line tools is obtained with the argument ``--help``, for example, executing the command ``./inverse --help`` inside the ``bin`` directory produces the following output:

    Inverse
    Usage: ./inverse [OPTIONS]

    Options:
    -h,--help                   Print this help message and exit
    -I,--inputImage TEXT        Input image filename
    -O,--outputImage TEXT       Output image filename
    -S,--show                   Display input and output images in new windows

### Unit tests

Each command line tool is associated to a unit test that can be executed with the tool ``bin/test``. For example the tool ``inverse`` can be tested with the command ``./test -P inverse``. If the test detects an error, you can ask to see an error map with the argument ``-S``: this will display an image showing where the errors are located: the brighter a pixel is, the further it is from the expected result. 

**Important:** The unit tests work by comparing the result of your functions to results produced with reference implementations. Implementation details, especially when working with floatting point values can lead to small disprecancies between similar results. While the test program tries to deal with this issue it can still detect false positives, i.e., say that a function is wrong while it is indeed right. In case of doubt call a professor to check your work.

As with any unit test, it is also possible that the unit tests suffer from false negatives, i.e., say that a program is correct while it is not.


## Project chosen :
NL_Means

[Introduction à la recherche](https://perso.esiee.fr/~perretb/I5FM/TAI/recherche/index.html) Alternative Link on Local [Introduction à la recherche](https://github.com/michel-ch/image-processing/blob/main/site/Introduction%20%C3%A0%20la%20recherche%20%E2%80%94%20Documentation%20Traitement%20et%20analyse%20d'images%201.html)

[Article](https://github.com/michel-ch/traitement-images/blob/main/image/NL_Means.pdf)

[Report](https://github.com/michel-ch/traitement-images/blob/main/image/Compte%20rendu%20NL%20Mean%20CHEN%20Michel.pdf)

[Code](https://github.com/michel-ch/traitement-images/tree/main/image/nlmeansC)

