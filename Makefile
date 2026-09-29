NVCC ?= nvcc
TARGET := signal_processing.exe
SOURCE := signal_processing.cu
NVCCFLAGS := -O2 -std=c++17 -lineinfo

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) $(SOURCE) -o $(TARGET)

run: $(TARGET)
	mkdir -p output
	./$(TARGET) --signals 256 --samples 8192 --window 9 \
		--output output/processed_samples.csv | tee output/execution.log

clean:
	rm -f $(TARGET)
	rm -f output/execution.log output/processed_samples.csv
