
#ifndef __DEFINES_H_
#define __DEFINES_H_

#define HASHTABLE_FREE -1
#define BOUNDARY 1
#define INTERIOR 2
#define BOTH 0

#define ISPOWER2(v) ((v) && !((v) & ((v) - 1)))
            
#define IDX3D(x,y,z,X,Y) ((z)*((Y)*(X)) + ((y)*(X)) + (x))

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

#define LOG(X) _LOG( X )
#define _LOG(X) _LOG_ ## X

#define _LOG_32 5
#define _LOG_16 4
#define _LOG_8  3
#define _LOG_4  2
#define _LOG_2  1
#define _LOG_1  0

#define NEIGHLIST_PACKSIZE 8
#define NEIGHLIST_PACKSIZE_LOG LOG(NEIGHLIST_PACKSIZE)
#define KERNEL_PACKSIZE 4

#define MAXNEIGHBORLISTSIZE 64

#define VECTOR_WIDTH 4

#endif
