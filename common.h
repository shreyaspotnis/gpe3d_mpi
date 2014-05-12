#ifndef COMMON_H
#define COMMON_H

/* Preprocessor definitions */
#define TRUE 1
#define FALSE 0
#define HBAR 1.05457173e-34
#define M_RB 1.44e-25
#define A_BG 5.1e-9
#define PI 3.1415926

#define MASTER_RANK 0
/* End of preprocessor definitions */

/* Globals */
extern int rank, size;
/* End of globals */

// POW2 returns 1 if v is a power of 2
#define POW2(v) (v && !(v & (v - 1)))

#endif
