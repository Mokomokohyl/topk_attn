NVCC ?= nvcc
ARCH ?= -arch=sm_120a
CXXFLAGS := -O3 -std=c++17 $(ARCH)
BIN_DIR := bin

# Source roots
BASE_ORIG := baselines/original/cuda
PLAN1 := baselines/variants/plan1/cuda
PLAN2 := baselines/variants/plan2/cuda

topk:
	$(NVCC) -O3 -std=c++17 $(ARCH) -o gemv_topk ./gemv_topk.cu && ./gemv_topk --verify 4
fd:
	$(NVCC) -O3 $(ARCH) -o flash_decoding flash_decoding.cu && ./flash_decoding

# Original baselines under baselines/original/cuda
s1:
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) -o $(BIN_DIR)/baseline_1 $(BASE_ORIG)/baseline_1.cu && $(BIN_DIR)/baseline_1
s2:
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) -o $(BIN_DIR)/baseline_2 $(BASE_ORIG)/baseline_2.cu && $(BIN_DIR)/baseline_2
s3:
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) -o $(BIN_DIR)/baseline_3 $(BASE_ORIG)/baseline_3.cu && $(BIN_DIR)/baseline_3

# Plan 1 variants
s1_p1:
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) -o $(BIN_DIR)/baseline_1_p1 $(PLAN1)/baseline_1.cu && $(BIN_DIR)/baseline_1_p1
s2_p1:
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) -o $(BIN_DIR)/baseline_2_p1 $(PLAN1)/baseline_2.cu && $(BIN_DIR)/baseline_2_p1
s3_p1:
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) -o $(BIN_DIR)/baseline_3_p1 $(PLAN1)/baseline_3.cu && $(BIN_DIR)/baseline_3_p1

# Plan 2 variants
s1_p2:
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) -o $(BIN_DIR)/baseline_1_p2 $(PLAN2)/baseline_1.cu && $(BIN_DIR)/baseline_1_p2
s2_p2:
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) -o $(BIN_DIR)/baseline_2_p2 $(PLAN2)/baseline_2.cu && $(BIN_DIR)/baseline_2_p2
s3_p2:
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) -o $(BIN_DIR)/baseline_3_p2 $(PLAN2)/baseline_3.cu && $(BIN_DIR)/baseline_3_p2

clean:
	@rm -f ./gemv_topk
	@rm -f ./baseline_1
	@rm -f ./baseline_2
	@rm -f ./baseline_3
	@rm -f ./flash_decoding
	@rm -f ./gemv_profile_split
	@rm -rf $(BIN_DIR)/*

.PHONY: gemv_topk fd s1 s2 s3 s1_p1 s2_p1 s3_p1 s1_p2 s2_p2 s3_p2 clean