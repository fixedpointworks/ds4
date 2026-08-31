#ifndef XH2RT_STATUS_H
#define XH2RT_STATUS_H

#include <stdint.h>

/* One private record per KLD: eight published stripe results followed by
 * one scratch word per hart. Only lane zero publishes its stripe result. */
#define XH2RT_STATUS_STRIPES 8u
#define XH2RT_STATUS_HARTS_PER_STRIPE 4u
#define XH2RT_STATUS_HARTS 32u
#define XH2RT_STATUS_BYTES 32u
#define XH2RT_STATUS_RECORD_BYTES 160u
#define XH2RT_STATUS_PENDING UINT32_MAX

#define XH2RT_KERNEL_OK 0u
#define XH2RT_KERNEL_INVALID_ARGUMENT 1u
#define XH2RT_KERNEL_PRIMITIVE_FAILURE 2u

#endif
