
#ifndef __DEFINES_H_
#define __DEFINES_H_

#define HASHTABLE_FREE -1
#define BOUNDARY 1
#define INTERIOR 2

/// The maximum number of atoms that can be stored in a link cell.
#define MAXATOMS 64 

#define WARP_SIZE		32

#define THREAD_ATOM_CTA         128
#define WARP_ATOM_CTA		128
#define CTA_CELL_CTA		128

// NOTE: the following is tuned for GK110
#ifdef DOUBLE
#define THREAD_ATOM_ACTIVE_CTAS 	10	// 62%
#define WARP_ATOM_ACTIVE_CTAS 		12	// 75%
#define CTA_CELL_ACTIVE_CTAS 		10	// 62%
#else
// 100% occupancy for SP
#define THREAD_ATOM_ACTIVE_CTAS 	16
#define WARP_ATOM_ACTIVE_CTAS 		16
#define CTA_CELL_ACTIVE_CTAS 		16
#endif

#define VECTOR_WIDTH 4

#endif
