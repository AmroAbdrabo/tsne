#include "Pool.h"
#include <stdlib.h>
#include <string.h>

namespace memory {

uint64_t Pool::dataSize;
void * Pool::data;
uint64_t Pool::remainingSize;
void * Pool::nextFreeData;
uint64_t Pool::lowerAddressBound;
uint64_t Pool::upperAddressBound;

void Pool::allocate(uint64_t size) {

	int result = posix_memalign((void **) &(data), 64, size);
	memset(data, 0, size);

	dataSize = size;
	remainingSize = size;
	nextFreeData = data;

	lowerAddressBound = (uint64_t) data;
	upperAddressBound = lowerAddressBound + size;

}

void* Pool::getMemory(uint64_t size) {

	void *memory = NULL;

	if (remainingSize >= size) {
		memory = nextFreeData;
		uint64_t aligned64Size = 0;
		if (((size >> 6) << 6) == size) {
			aligned64Size = size;
		} else {
			aligned64Size = (size + 64) & (~0x3F);
		}
		nextFreeData = (void*) (((uint64_t) nextFreeData) + aligned64Size);
		remainingSize -= aligned64Size;
	} else {
		int result = posix_memalign((void **) &(memory), 64, size);
	}

	return memory;

}

void Pool::free(void* memory) {
	if ((((uint64_t) memory) < lowerAddressBound) || (((uint64_t) memory) >= upperAddressBound)) {
		free(memory);
	}
}

void Pool::freeAll() {
	free(data);
}

void Pool::reset() {
	remainingSize = dataSize;
	nextFreeData = data;
}

} /* namespace memory */
