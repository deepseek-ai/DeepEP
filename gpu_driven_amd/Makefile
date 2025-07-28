ABS_ROOT_PATH := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

RCCL_HOME:=../thirdparty/rccl
HIP_HOME?=/opt/rocm
CONDA_LIB_HOME?=/usr/lib

INC = -I./ -I../include -I$(HIP_HOME)/include -I${CONDA_LIB_HOME}/../include -L${CONDA_LIB_HOME}
LIBS = -lz -lelf -libverbs -L ${HIP_HOME}/lib -lamdhip64
LIBS_SHARED = -lglog -lgflags -lgtest -lz -lelf -libverbs
override CXXFLAGS += -O3 -g -std=c++17 -Wno-pointer-arith -Wno-interference-size -fPIC -D__HIP_PLATFORM_AMD__
DEPS = *.h
PLUGIN_SO = librccl-net-uccl.so
NCCL_INC:= -I$(RCCL_HOME)/build/release/include -I$(RCCL_HOME)/src/include -I$(HIP_HOME)/include

# HIP compiler
HIPCC := $(HIP_HOME)/bin/hipcc

# Update include paths for our HIP headers
INCLUDES := -Iinclude -I$(HIP_HOME)/include -I/usr/include -I../include -I../dependencies

# Source files - update extensions for HIP
SRC_CPP := src/proxy.cpp src/rdma.cpp src/common.cpp src/peer_copy_worker.cpp
SRC_HIP := src/gpu_kernel.hip src/peer_copy.hip
SRC_LOCAL := bench/benchmark_local.hip
SRC_REMOTE := bench/benchmark_remote.hip

# Object files
OBJ_CPP := $(SRC_CPP:.cpp=.o)
OBJ_HIP := $(SRC_HIP:.hip=.o)

OBJ_LOCAL := $(OBJ_CPP) $(OBJ_HIP) $(SRC_LOCAL:.hip=.o)
OBJ_REMOTE := $(OBJ_CPP) $(OBJ_HIP) $(SRC_REMOTE:.hip=.o)

# Library source files
lib_src = $(wildcard *.cc)
lib_src := $(filter-out %_main.cc,$(lib_src))
lib_src := $(filter-out %_test.cc,$(lib_src))
lib_src := $(filter-out %_plugin.cc,$(lib_src))
lib_obj = $(lib_src:.cc=.o)

# Test source files
test_src = $(wildcard *_test.cc)
test_src := $(filter-out rdma_test.cc,$(test_src))
test_bin = $(test_src:.cc=)

# Targets
TARGET_LOCAL := benchmark_local
TARGET_REMOTE := benchmark_remote

.PHONY: build
build: $(test_bin) $(lib_obj) $(PLUGIN_SO) librdma_hip.a $(TARGET_LOCAL) $(TARGET_REMOTE)

# Test binary compilation
%_test: %_test.cc $(DEPS) $(lib_obj)
	g++ $< -o $@ $(lib_obj) $(INC) $(LIBS) $(CXXFLAGS)

# C++ object file compilation
%.o: %.cc $(DEPS)
	g++ -c $< -o $@ $(INC) $(CXXFLAGS)

# C++ source compilation for our files
%.o: %.cpp
	g++ $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

# HIP compilation rule
%.o: %.hip
	$(HIPCC) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

# Plugin shared library
$(PLUGIN_SO): nccl_plugin.cc $(DEPS) $(lib_obj)
	g++ $(NCCL_INC) -fPIC -shared -o $@ -Wl,-soname,$(PLUGIN_SO) nccl_plugin.cc $(lib_obj) $(INC) $(LIBS_SHARED) $(CXXFLAGS)

# Static library
librdma_hip.a: $(lib_obj)
	ar rcs $@ $(lib_obj)

# Benchmark targets - linking with HIP
$(TARGET_LOCAL): $(OBJ_LOCAL)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES) $(OBJ_LOCAL) $(LIBS) -o $@

$(TARGET_REMOTE): $(OBJ_REMOTE)
	$(HIPCC) $(CXXFLAGS) $(INCLUDES) $(OBJ_REMOTE) $(LIBS) -o $@

# Clean all generated files
.PHONY: clean
clean:
	rm -f *.o $(test_bin) $(PLUGIN_SO) librdma_hip.a $(TARGET_LOCAL) $(TARGET_REMOTE) *.d src/*.d bench/*.d

# Automatically include dependency files if they exist
-include $(OBJ_LOCAL:.o=.d) $(OBJ_REMOTE:.o=.d)