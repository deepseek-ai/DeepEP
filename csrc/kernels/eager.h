
#ifndef __EAGER_SUPPORT_H__
#define __EAGER_SUPPORT_H__

namespace deep_ep {

#include <stdint.h>

/*
 * @brief IB MTU based Tagging Layout
 *
 * MTU-0    [0, ..., 4079, t0, t1, t2, ..., t15], total 4096 bytes
 * ...
 * MTU-last [0, ..., last, t0, t1, t2, ..., t15], total last + 1 + 16 bytes, note (last + 1) % 16 == 0
 */


// use IB MTU as segment size if PCIe Relaxed Ordering is Off, max IB MTU is 4096 bytes
#define PCIE_SEG_LEN_LOG (12)
#define PCIE_SEG_LEN (1 << PCIE_SEG_LEN_LOG)
#define PCIE_SEG_LEN_MASK (PCIE_SEG_LEN - 1)

#define AR_MSG_ALIGNMENT (1 << 4)
// make TLP dst aligned by 256 bytes, avoid TLP spliting
#define AR_MSG_LONG_ALIGNMENT (1 << 8)

// for int4 alignment
#define PCIE_TAIL_SZ (1 << 4)

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define CEIL_ALIGN(a, b) (CEIL_DIV(a, b) * (b))
#define NON_ZERO(x, y) ((x) < (y) ? (0) : ((x) - (y)))
#define PAGE_N(size) CEIL_DIV(size, PCIE_SEG_LEN - PCIE_TAIL_SZ)
#define FULL_MSG_LEN(size) ((size) + (PAGE_N(size) * PCIE_TAIL_SZ))
#define EXTEND_FOR_TAG_AND_ALIGN(size, alignment) CEIL_ALIGN(FULL_MSG_LEN(size), alignment)

#define DISPATCH_ROUND_INT 0x40000000
#define COMBINE_ROUND_INT 0xc0000000
#define ROUND_MASK 0x3fffffff

#define SWITCH_EAGER(inner_macro, ...) \
do { \
    if (use_eager) { \
        inner_macro(true, __VA_ARGS__); \
    } else {\
        inner_macro(false, __VA_ARGS__); \
    }\
} while (0); break;

};

#endif // __AR_SUPPORT_H__