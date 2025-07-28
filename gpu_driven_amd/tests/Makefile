HIP_HOME?=/opt/rocm
EFA_HOME:=/opt/amazon/efa
HIPCC=${HIP_HOME}/bin/hipcc

CXXFLAGS = -I${EFA_HOME}/include -L${EFA_HOME}/lib -I$(HIP_HOME)/include -L ${HIP_HOME}/lib -libverbs -lamdhip64 -lpthread -std=c++17 -D__HIP_PLATFORM_AMD__
HIPFLAGS = --offload-arch=gfx90a # for MI200 series, adjust for your GPU

HIP_BIN = pcie_bench gpu_to_cpu_bench batched_gpu_to_cpu_bench
all: $(HIP_BIN)

pcie_bench: pcie_bench.hip
	${HIPCC} -O3 -g pcie_bench.hip -o pcie_bench $(CXXFLAGS) $(HIPFLAGS)

gpu_to_cpu_bench: gpu_to_cpu_bench.hip
	${HIPCC} -O3 -g gpu_to_cpu_bench.hip -o gpu_to_cpu_bench $(CXXFLAGS) $(HIPFLAGS)

batched_gpu_to_cpu_bench: batched_gpu_to_cpu_bench.hip
	${HIPCC} -O3 -g batched_gpu_to_cpu_bench.hip -o batched_gpu_to_cpu_bench $(CXXFLAGS) $(HIPFLAGS)

clean:
	rm -f $(HIP_BIN)

.PHONY: all clean