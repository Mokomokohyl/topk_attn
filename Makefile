topk:
	nvcc -O3 -std=c++17 -arch=sm_120a -o gemv_topk ./gemv_topk.cu && ./gemv_topk --verify 4
fd:
	nvcc -O3 -arch=sm_120a -o flash_decoding flash_decoding.cu && ./flash_decoding
s1:
	nvcc -O3 -std=c++17 -arch=sm_120a -o baseline_1 baseline_1.cu && ./baseline_1
s2:
	nvcc -O3 -std=c++17 -arch=sm_120a -o baseline_2 baseline_2.cu && ./baseline_2
s3:
	nvcc -O3 -std=c++17 -arch=sm_120a -o baseline_3 baseline_3.cu && ./baseline_3
clean:
	@rm -f ./gemv_topk
	@rm -f ./baseline_1
	@rm -f ./baseline_2
	@rm -f ./baseline_3
	@rm -f ./flash_decoding
	@rm -f ./gemv_profile_split

.PHONY: gemv_topk fd s1 s2 clean