cmake_minimum_required(VERSION 3.5)
project(PointnetInference LANGUAGES CXX)
if (DEFINED ENV{ONEAPI_ROOT})
  if(NOT WIN32)
    set(CMAKE_CXX_COMPILER icpx)
    set(CMAKE_C_COMPILER icx)
else()
    set(CMAKE_CXX_COMPILER icx)
endif()
endif()

add_executable(pointnet pointnet.cpp)

find_package(DNNL REQUIRED)

if(DEFINED ENV{ONEAPI_ROOT})
message("-- DNNL found:" ${DNNL_FOUND})
  if( ${DNNL_FOUND} )
    target_link_libraries(pointnet PUBLIC DNNL::dnnl)
    target_include_directories(pointnet PUBLIC $ENV{DNNLROOT}/share/doc/dnnl/examples/)
    target_compile_options(pointnet PUBLIC -fsycl -fsycl-targets=spir64)
  endif()
else()
if(${DNNL_FOUND} AND DEFINED DNNL_BASE_DIR)
    target_link_libraries(pointnet PUBLIC DNNL::dnnl)
    # Copy utils from oneDNN github repo to include directory
    file(COPY ${DNNL_BASE_DIR}/examples/example_utils.hpp DESTINATION ${DNNL_DIR}/../../../include)
    target_include_directories(pointnet PUBLIC "${DNNL_DIR}/../../../include")
  endif()
endif()

target_compile_options(pointnet PUBLIC -fsycl)

if(DEFINED TARGET_ARCH)
  # Specify full target architecture, to increase portability e.g. AMD GPUs
  target_compile_options(pointnet PUBLIC -fsycl-targets=${TARGET_ARCH})
elseif(${TARGET} MATCHES "NVIDIA")
  target_compile_options(pointnet PUBLIC -fsycl-targets=nvptx64-nvidia-cuda)
elseif(${TARGET} MATCHES "INTEL")
  target_compile_options(pointnet PUBLIC -fsycl-targets=spir64)
endif()

target_link_libraries(pointnet PRIVATE sycl )
