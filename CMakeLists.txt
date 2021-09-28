cmake_minimum_required(VERSION 3.16)
project(sparsedbn)

set(CMAKE_CXX_STANDARD 14)

if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -msse2")
    add_compile_definitions(EIGEN_NO_DEBUG)
    if (MKL_ROOT)
      message("MKL BUILD")
      set(MKL_COMPILE_LINE -DMKL_ILP64 -m64 -I/usr/include/mkl/ -fopenmp)
      add_compile_definitions(EIGEN_USE_MKL_ALL)
      link_directories(${MKL_ROOT})
      link_libraries(mkl_intel_ilp64 mkl_gnu_thread mkl_core gomp pthread m dl)
    endif()
endif()

find_package(Eigen3 3.3 REQUIRED)
find_library(LIBLBFGS lbfgs)
find_library(LIBFLATBUFFERS flatbuffers)
find_path(LIBFLATBUFFERS_INCLUDE_DIR flatbuffers)
find_package(OpenSSL REQUIRED)
find_package(Boost QUIET)

message(${EIGEN3_INCLUDE_DIR})
message(${LIBFLATBUFFERS_INCLUDE_DIR})

include_directories(${LIBFLATBUFFERS_INCLUDE_DIR})
include_directories("/usr/local/include")
include_directories("/usr/include")
include_directories("/home/fatherly_pelican/software/include")

if (MPI_BUILD)
    find_package(MPI REQUIRED)
endif()

SET(BASEPATH "${CMAKE_SOURCE_DIR}")
INCLUDE_DIRECTORIES("${BASEPATH}")

set(REGULARIZED_NETWORK_SOURCES
        regularized_network/BNNodeTrainer.cpp
        regularized_network/BNNodeTrainer.h
        regularized_network/dataset.cpp
        regularized_network/dataset.h
        regularized_network/DBNNodeTrainer.cpp
        regularized_network/DBNNodeTrainer.h
        regularized_network/DynamicPathTrainResult.h
        regularized_network/FixedPathTrainResult.h
        regularized_network/NodeTrainer.cpp
        regularized_network/NodeTrainer.h
        regularized_network/PathTrainer.cpp
        regularized_network/PathTrainer.h
        regularized_network/PathTrainResult.h
        regularized_network/local/BNPathTrainer.cpp
        regularized_network/local/BNPathTrainer.h
        regularized_network/local/DBNPathTrainer.cpp
        regularized_network/local/DBNPathTrainer.h)

set(REGULARIZED_NETWORK_DISTRIBUTED_SOURCES
        regularized_network/distributed/BNPathTrainer.cpp
        regularized_network/distributed/BNPathTrainer.h
        regularized_network/distributed/DBNPathTrainer.cpp
        regularized_network/distributed/DBNPathTrainer.h
        regularized_network/distributed/Worker.h
        regularized_network/distributed/WorkerRequest.cpp
        regularized_network/distributed/WorkerRequest.h
        regularized_network/distributed/WorkerResponse.h
        regularized_network/distributed/WorkPool.cpp
        regularized_network/distributed/WorkPool.h)

set(COMMON_SOURCES
        common/train.cpp
        common/train.h
        common/instrumentation.cpp
        common/instrumentation.h
        common/cxxopts.hpp
        common/digraph.cpp
        common/HillClimbingHotStart.h
        common/digraph.h
        common/io.cpp
        common/io.h common/HotStart.h
        common/container_hasher.hpp)


set(HILLCLIMBER_SOURCES
        hillclimber/hill.h
        hillclimber/hill.cpp)

set(ADAPTATIVE_SOURCES
        adaptative/MultinomialLogisticRegression.cpp
        adaptative/MultinomialLogisticRegression.h
        adaptative/weights.h
        adaptative/weights.cpp)

add_executable(hill
        ${HILLCLIMBER_SOURCES}
        ${COMMON_SOURCES}
        hillclimber/main.cpp
        regularized_network/dataset.cpp
        regularized_network/dataset.h
        common/cxxopts.hpp)

target_link_libraries(hill Eigen3::Eigen)
target_link_libraries(hill OpenSSL::Crypto)

add_executable(dbntrain
        dbntrain/main.cpp
        dbntrain/train.cpp
        dbntrain/train.h
        ${REGULARIZED_NETWORK_SOURCES}
        ${HILLCLIMBER_SOURCES}
        ${ADAPTATIVE_SOURCES}
        ${COMMON_SOURCES})

target_link_libraries(dbntrain Eigen3::Eigen)
target_link_libraries(dbntrain ${LIBLBFGS})
target_link_libraries(dbntrain ${LIBFLATBUFFERS})
target_link_libraries(dbntrain OpenSSL::Crypto)

add_executable(bntrain
        bntrain/main.cpp
        bntrain/train.cpp
        bntrain/train.h
        ${REGULARIZED_NETWORK_SOURCES}
        ${HILLCLIMBER_SOURCES}
        ${ADAPTATIVE_SOURCES}
        ${COMMON_SOURCES})

target_link_libraries(bntrain Eigen3::Eigen)
target_link_libraries(bntrain ${LIBLBFGS})
target_link_libraries(bntrain ${LIBFLATBUFFERS})
target_link_libraries(bntrain OpenSSL::Crypto)

if(MKL_COMPILE_LINE)
  target_compile_options(dbntrain PUBLIC "${MKL_COMPILE_LINE}")
  target_compile_options(hill PUBLIC "${MKL_COMPILE_LINE}")
  target_compile_options(bntrain PUBLIC "${MKL_COMPILE_LINE}")
endif()

if(MKL_LINK_LINE)
  target_link_options(dbntrain PUBLIC ${MKL_LINK_LINE})
  target_link_options(hill PUBLIC ${MKL_LINK_LINE})
  target_link_options(bntrain PUBLIC ${MKL_LINK_LINE})

  target_link_libraries(dbntrain ${MKL_ROOT}/lib/intel64/libmkl_rt.so)
  target_link_libraries(bntrain ${MKL_ROOT}/lib/intel64/libmkl_rt.so)
  target_link_libraries(hill ${MKL_ROOT}/lib/intel64/libmkl_rt.so)
endif()

if (MPI_BUILD)
    include_directories(${MPI_INCLUDE_PATH})
    add_compile_definitions(MPI_BUILD)

    target_sources(dbntrain PRIVATE ${REGULARIZED_NETWORK_DISTRIBUTED_SOURCES})
    target_link_libraries(dbntrain ${MPI_LIBRARIES})

    target_sources(bntrain PRIVATE ${REGULARIZED_NETWORK_DISTRIBUTED_SOURCES})
    target_link_libraries(bntrain ${MPI_LIBRARIES})

    if(MPI_COMPILE_FLAGS)
        set_target_properties(dbntrain PROPERTIES
                COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
        set_target_properties(bntrain PROPERTIES
                COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    endif()

    if(MPI_LINK_FLAGS)
        set_target_properties(dbntrain PROPERTIES
                LINK_FLAGS "${MPI_LINK_FLAGS}")
        set_target_properties(bntrain PROPERTIES
                LINK_FLAGS "${MPI_LINK_FLAGS}")
    endif()
endif()

if(Boost_FOUND)
    add_compile_definitions(BOOST_FOUND)
    include_directories(${Boost_INCLUDE_DIR})
    link_directories(${Boost_LIBRARY_DIR})
    target_link_libraries(hill ${Boost_CONTAINER_HASH_LIBRARY})
    target_link_libraries(dbntrain ${Boost_CONTAINER_HASH_LIBRARY})
    target_link_libraries(bntrain ${Boost_CONTAINER_HASH_LIBRARY})
endif()

get_target_property(MAIN_CFLAGS bntrain LINK_OPTIONS)
# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags are: ${MAIN_CFLAGS}")
