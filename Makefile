# =============================================================================
# POINT — Ultra-Fast Hybrid GPU + CPU Distinguished Point (100% TAME) Generator
# Compatible 100% with PSCKangaroo v56C & massive_tames.bin
# Integrates BMI2/ADX Assembly + DRS62 Batch Inversion from tames-gen-main
# =============================================================================

CUDA_HOME ?= /usr
NVCC ?= nvcc
CC = g++
CC_C = gcc

# Multi-architecture support:
#   sm_86 = RTX 3060/3070/3080/3090 (Ampere)
#   sm_89 = RTX 4060/4070/4080/4090 (Ada Lovelace)
GPU_ARCH ?= -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89

V46_ENABLE ?= 0
V46_EXPLORER_PCT ?= 20
V46_EXPLORER_SHIFT ?= 2

V45_OCCUPANCY ?= 1
V45_TABLE_BITS ?= 33
V45_PNT_GROUP_CNT ?= 32
V45_STEP_CNT ?= 1000

V46_DEFS = -DV46_ENABLE=$(V46_ENABLE) \
           -DV46_EXPLORER_PCT=$(V46_EXPLORER_PCT) \
           -DV46_EXPLORER_SHIFT=$(V46_EXPLORER_SHIFT)

V45_DEFS = -DV45_OCCUPANCY=$(V45_OCCUPANCY) \
           -DV45_TABLE_BITS=$(V45_TABLE_BITS) \
           -DV45_PNT_GROUP_CNT=$(V45_PNT_GROUP_CNT) \
           -DV45_STEP_CNT=$(V45_STEP_CNT)

ALL_DEFS = $(V46_DEFS) $(V45_DEFS)

NVCC_FLAGS = -O3 $(GPU_ARCH) $(ALL_DEFS) -Xcompiler -O3,-march=native,-pthread --ptxas-options=-v
CC_FLAGS = -O3 -march=native -mavx2 -funroll-loops -pthread $(ALL_DEFS)
C_FLAGS  = -O3 -march=native -mavx2 -funroll-loops

LIBS = -lcuda -lcudart -lpthread

OBJS = point_main.o Ec.o GpuKang.o RCGpuCore.o utils.o G_stub.o

TARGET = point

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(OBJS) $(LIBS)
	@echo ""
	@echo "=== Build complete: ./$(TARGET) (Hybrid GPU + CPU 100% TAME Ultra-Fast Generator) ==="
	@echo ""

point_main.o: point_main.cpp defs.h utils.h GpuKang.h TameStore.h Ec.h G.h fast_inv.h
	$(CC) $(CC_FLAGS) -I$(CUDA_HOME)/include -c point_main.cpp -o $@

G_stub.o: G_stub.c G.h
	$(CC_C) $(C_FLAGS) -c G_stub.c -o $@

Ec.o: Ec.cpp Ec.h defs.h
	$(CC) $(CC_FLAGS) -c Ec.cpp -o $@

GpuKang.o: GpuKang.cpp GpuKang.h defs.h utils.h Ec.h TameStore.h
	$(CC) $(CC_FLAGS) -I$(CUDA_HOME)/include -c GpuKang.cpp -o $@

RCGpuCore.o: RCGpuCore.cu defs.h RCGpuUtils.h
	$(NVCC) $(NVCC_FLAGS) -c RCGpuCore.cu -o $@

utils.o: utils.cpp utils.h defs.h
	$(CC) $(CC_FLAGS) -c utils.cpp -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
