// Couette flow 
// Multi-block Lattice Boltzmann Method.
// The source code is written in c programming language.
// The source code is written in C programming language.
// Copyright @2021 Takeshi Seta All Rights Reserved.
//
// 6  2  5
//    |
// 3--0--1
//    |
// 7  4  8
//
// m   : grid spacing ratio 
//
// rho : density (in the whole region)
// rhoc: density (in the coarse grid)
// rhof: density (in the fine grid)
//
// u   : horizontal velocity component (in the whole region, time = n)
// uc  : horizontal velocity component (in the coarse grid, time = n)
// uf  : horizontal velocity component (in the fine grid, time = n)
//
// v   : vertical velocity component (in the whole region, time = n)
// vc  : vertical velocity component (in the coarse grid, time = n)
// vf  : vertical velocity component (in the fine grid, time = n)
//
// ucn : horizontal velocity component (in the coarse grid, time = n -1)
// ufn : horizontal velocity component (in the fine grid, time = n -1)
// vcn : vertical velocity component (in the coarse grid, time = n - 1)
// vfn : vertical velocity component (in the fine grid, time = n - 1)
//
// ut  : horizontal velocity component of the top wall
// vt  : vertical velocity component of the top wall
// ub  : horizontal velocity component of the bottom wall
// vb  : vertical velocity component of the bottom wall
//
// nx  : number of grid points (x-axis, in the whole region)
// ny  : number of grid points (y-axis, in the whole region)
// nxc : number of grid points (x-axis, in the coarse grid)
// nyc : number of grid points (y-axis, in the coarse grid)
// nxf : number of grid points (x-axis, in the fine grid)
// nyf : number of grid points (y-axis, in the fine grid)
//
// fc  : distribution function (in the coarse grid)
// fc0 : equilibrium distribution function (in the coarse grid)
// ff  : distribution function (in the fine grid)
// ff0 : equilibrium distribution function (in the fine grid)
//
// cx  : discrete velocity (x-axis)
// cy  : discrete velocity (y-axis)
// tauc: relaxation time (in the coarse grid)
// tauf: relaxation time (in the fine grid)
// nu  : kinematic viscosity (in the whole region)
// nuf : kinematic viscosity (in the whole region)
// re  : Reynolds number
//
// ftc : distribution function on the interface for the coarse grid
// ftc0: equilibrium distribution function on the interface
//       for the coarse grid
//
// fch : distribution function on the interface
//       for the Lagrangian interpolation  (time = n - 1/2)
// ctf3: distribution function on the interface
//       for the Lagrangian interpolation   (time = n )
// ctf2: distribution function on the interface
//       for the Lagrangian interpolation   (time = n - 1)
// ctf1: distribution function on the interface
//       for the Lagrangian interpolation   (time = n - 2)
//
// fc0h: equilibrium distribution function on the interface
//       for the Lagrangian interpolation  (time = n - 1/2)
// ctf03: equilibrium distribution function on the interface
//        for the Lagrangian interpolation  (time = n )
// ctf02: equilibrium distribution function on the interface
//        for the Lagrangian interpolation  (time = n - 1)
// ctf01: equilibrium distribution function on the interface
//        for the Lagrangian interpolation  ( time = n - 2)
// a, b, c, d : coefficient for the spline interpolation


#include <stdio.h>
#include <math.h>
#include <stdlib.h>

# define DIM 50

int main(void)
{
  FILE   *fp;
  int    nx = 20, ny = 32, nxc, nyc, nxf, nyf, time = 0, loop1, loop2;
  int    i, j, k, m, n, kk, in, jn, nb, im, ip;
  double  u[DIM][DIM],  v[DIM][DIM], rho[DIM][DIM];
  double uc[DIM][DIM], vc[DIM][DIM], ucn[DIM][DIM], vcn[DIM][DIM], rhoc[DIM][DIM];
  double uf[DIM][DIM], vf[DIM][DIM], ufn[DIM][DIM], vfn[DIM][DIM], rhof[DIM][DIM];
  double fc[9][DIM][DIM], fc0[9][DIM][DIM], ftmp[9][DIM][DIM], cx[9], cy[9];
  double ff[9][DIM][DIM], ff0[9][DIM][DIM];

  double    ftc[9][DIM],  ftc0[9][DIM];
  double   ctf1[9][DIM],  ctf2[9][DIM],  ctf3[9][DIM],  fch[9][DIM];
  double  ctf01[9][DIM], ctf02[9][DIM], ctf03[9][DIM], fc0h[9][DIM];
  double   a[DIM], b[DIM], c[DIM], d[DIM], mm[DIM][DIM];

  double tmp, u2, nu, norm, nuf, r, ut = 0.01, vt = 0.0, ub = 0.0, vb = 0.0;
  double tauc = 1.2, tauf;
  int dun;
  char   aa[DIM][DIM];

  m = 2;
  nxf = m*nx; nyf = ny;
  nxc =   nx; nyc = ny/m + 1;
  tauf = 0.5 + (double)m*(tauc -0.5);

  // initial condition
  nu = (tauc - 0.5)/3.0;
  nuf = (tauf - 0.5)/3.0/(double)m;
  printf("nu = %8.6e(%6.4e)\n", nu, nuf);
  printf("nx , ny  = %d, %d\n", nx , ny );
  printf("nxc, nyc = %d, %d\n", nxc, nyc);
  printf("nxf, nyf = %d, %d\n", nxf, nyf);
//  scanf("%d", &dun);

  cx[0] = 0.0; cy[0] = 0.0;
  cx[1] = 1.0; cy[1] = 0.0;  cx[2] = 0.0; cy[2] = 1.0;
  cx[3] =-1.0; cy[3] = 0.0;  cx[4] = 0.0; cy[4] =-1.0;
  cx[5] = 1.0; cy[5] = 1.0;  cx[6] =-1.0; cy[6] = 1.0;
  cx[7] =-1.0; cy[7] =-1.0;  cx[8] = 1.0; cy[8] =-1.0;

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
     u[i][j] = 0.0;  v[i][j] = 0.0;  rho[i][j] = 1.0;
  } }

  for(i = 0; i <= nxc; i++){ for(j = 0; j <= nyc; j++){
    uc[i][j] = 0.0; vc[i][j] = 0.0; ucn[i][j] = 0.0; vcn[i][j] = 0.0; rhoc[i][j] = 1.0;
  } }

  for(i = 0; i <= nxf; i++){ for(j = 0; j <= nyf; j++){
    uf[i][j] = 0.0; vf[i][j] = 0.0; ufn[i][j] = 0.0; vfn[i][j] = 0.0; rhof[i][j] = 1.0;
  } }

  for(i = 0; i <= nxc; i++){ for(j = 0; j <= nyc; j++){
    u2 = uc[i][j]*uc[i][j] + vc[i][j]*vc[i][j];      
    fc0[0][i][j] = rhoc[i][j]*(1.0 -3.0/2.0*u2)*4.0/9.0;
    for(k = 1; k <= 4; k++){
      tmp = cx[k]*uc[i][j] + cy[k]*vc[i][j];      
      fc0[k][i][j] = rhoc[i][j]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/9.0;
    }
    for(k = 5; k <= 8; k++){
      tmp = cx[k]*uc[i][j] + cy[k]*vc[i][j];      
      fc0[k][i][j] = rhoc[i][j]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/36.0;
    }
  } }

  for(i = 0; i <= nxc; i++){ for(j = 0; j <= nyc; j++){ for(k = 0; k <= 8; k++){
    fc[k][i][j] = fc0[k][i][j];
  } } }

//
  for(i = 0; i <= nxf; i++){ for(j = 0; j <= nyf; j++){
    u2 = uf[i][j]*uf[i][j] + vf[i][j]*vf[i][j];      
    ff0[0][i][j] = rhof[i][j]*(1.0 -3.0/2.0*u2)*4.0/9.0;
    for(k = 1; k <= 4; k++){
      tmp = cx[k]*uf[i][j] + cy[k]*vf[i][j];      
      ff0[k][i][j] = rhof[i][j]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/9.0;
    }
    for(k = 5; k <= 8; k++){
      tmp = cx[k]*uf[i][j] + cy[k]*vf[i][j];      
      ff0[k][i][j] = rhof[i][j]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/36.0;
    }
  } }

  for(i = 0; i <= nxf; i++){ for(j = 0; j <= nyf; j++){ for(k = 0; k <= 8; k++){
    ff[k][i][j] = ff0[k][i][j];
  } } }

//.. Lagrangian interpolation (initial condition)
  for(i = 0; i <= nxc; i++){ for(k = 0; k <= 8; k++){
     ctf1[k][i] = fc[k][i][1];
     ctf2[k][i] = fc[k][i][1];
     ctf3[k][i] = fc[k][i][1];
    ctf01[k][i] = fc[k][i][1];
    ctf02[k][i] = fc[k][i][1];
    ctf03[k][i] = fc[k][i][1];
  } }
  for(i = 0; i <= nxf; i++){ for(k = 0; k <= 8; k++){
     ftc[k][i] =  ff[k][i][nyf - 2];
    ftc0[k][i] = ff0[k][i][nyf - 2];
  } }

//calculation
  for(loop1 = 0; loop1 < 30; loop1++){
  for(loop2 = 0; loop2 < 100; loop2++){
    time++;

    for(i = 0; i <= nxc; i++){ for(j = 0; j <= nyc; j++){
      ucn[i][j] = uc[i][j]; vcn[i][j] = vc[i][j];
    } }
    for(i = 0; i <= nxf; i++){ for(j = 0; j <= nyf; j++){
      ufn[i][j] = uf[i][j]; vfn[i][j] = vf[i][j];
    } }

// Coase-grid
  // collision
    for(i = 0; i <= nxc; i++){ for(j = 0; j <= nyc; j++){
      u2 = uc[i][j]*uc[i][j] + vc[i][j]*vc[i][j];      
      fc0[0][i][j] = rhoc[i][j]*(1.0 -3.0/2.0*u2)*4.0/9.0;
      for(k = 1; k <= 4; k++){
        tmp = cx[k]*uc[i][j] + cy[k]*vc[i][j];      
        fc0[k][i][j] = rhoc[i][j]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/9.0;
      }  
      for(k = 5; k <= 8; k++){
        tmp = cx[k]*uc[i][j] + cy[k]*vc[i][j];      
        fc0[k][i][j] = rhoc[i][j]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/36.0;
      }
    } }

    for(i = 0; i <= nxc; i++){ for(j = 0; j <= nyc; j++){ for(k = 0; k <= 8; k++){
      fc[k][i][j] = fc[k][i][j] - (fc[k][i][j] - fc0[k][i][j])/tauc;
    } } }

//.. Lagrangian interpolation
    for(i = 0; i <= nxc; i++){ for(k = 0; k <= 8; k++){
       ctf1[k][i] = ctf2[k][i] ; ctf01[k][i] = ctf02[k][i];
       ctf2[k][i] = ctf3[k][i] ; ctf02[k][i] = ctf03[k][i];
       ctf3[k][i] = fc[k][i][1]; ctf03[k][i] = fc0[k][i][1];
    } }

    // propagation 
    for(i = 0; i <= nxc; i++){ for(j = 0; j <= nyc; j++){ for(k = 0; k <= 8; k++){
      ftmp[k][i][j] = fc[k][i][j];
    } } }

    for(i = 0; i <= nxc; i++){ for(j = 0; j <= nyc; j++){
      in = i + 1; if(i == nxc){ in =  0;}
      fc[1][in][j ] = ftmp[1][i][j];
    } }
    for(i = 0; i <= nxc; i++){ for(j = 0; j <  nyc; j++){
      jn = j + 1;
      fc[2][i][jn] = ftmp[2][i][j];
    } }
    for(i = 0; i <= nxc; i++){ for(j = 0; j <= nyc; j++){
      in = i - 1; if(i ==   0) {in = nxc;}
      fc[3][in][j] = ftmp[3][i][j];
    } }
    for(i = 0; i <= nxc; i++){ for(j = 1; j <= nyc; j++){
      jn = j - 1;
      fc[8][i][jn] = ftmp[4][i][j];
    } }
    for(i = 0; i <= nxc; i++){ for(j = 0; j <  nyc; j++){
      in = i + 1; if(i == nxc){ in =  0;}
      jn = j + 1;
      fc[5][in][jn] = ftmp[5][i][j]; 
    } }
    for(i = 0; i <= nxc; i++){ for(j = 0; j <  nyc; j++){
      in = i - 1; if(i ==   0) {in = nxc;}
      jn = j + 1;
      fc[6][in][jn] = ftmp[6][i][j];
    } }
    for(i = 0; i <= nxc; i++){ for(j = 1; j <=  nyc; j++){
      in = i - 1; if(i ==   0) {in = nxc;}
      jn = j - 1;
      fc[7][in][jn] = ftmp[7][i][j]; 
    } }
    for(i = 0; i <= nxc; i++){ for(j = 1; j <= nyc; j++){
      in = i + 1; if(i == nxc){ in =  0;}
      jn = j - 1;
      fc[8][in][jn] = ftmp[8][i][j]; 
    } }

    // bounce back condition
    // Non-equilibrium bounce back (Zou)
      for(i = 0; i <= nxc; i++){
        rhoc[i][nyc] = (fc[0][i][nyc] + fc[1][i][nyc] + fc[3][i][nyc]
                 + 2.0*(fc[2][i][nyc] + fc[5][i][nyc] + fc[6][i][nyc]))/(1.0 + vt);
        fc[4][i][nyc] = fc[2][i][nyc] - rhoc[i][nyc]*vt*2.0/3.0;
        fc[7][i][nyc] = fc[5][i][nyc] + 0.5*(fc[1][i][nyc] - fc[3][i][nyc])
                      - rhoc[i][nyc]*( ut/2.0 + vt/6.0);
        fc[8][i][nyc] = fc[6][i][nyc] - 0.5*(fc[1][i][nyc] - fc[3][i][nyc])
                      - rhoc[i][nyc]*(-ut/2.0 + vt/6.0);
      } 
//.. Interface (Fine --> Coase)
    for(i = 0; i <= nxc; i++){ for(k = 0; k <= 8; k++){
      in  = 2*i;
      fc[k][i][0] = ftc0[k][in]
      + (ftc[k][in] - ftc0[k][in]) * (double)m * (tauc - 1.0)/(tauf - 1.0);
    } }

    // physics
    for(i = 0; i <= nxc; i++){ for(j = 0; j <= nyc; j++){
      rhoc[i][j] = fc[0][i][j]; uc[i][j] = 0; vc[i][j] = 0;
      for( k = 1; k <= 8; k++){
        rhoc[i][j] = rhoc[i][j] + fc[k][i][j];
          uc[i][j] =   uc[i][j] + fc[k][i][j]*cx[k];
          vc[i][j] =   vc[i][j] + fc[k][i][j]*cy[k];
      } 
      uc[i][j] = uc[i][j]/rhoc[i][j];
      vc[i][j] = vc[i][j]/rhoc[i][j];
    } }

// Fine-grid
    for(n = 0; n < m; n++){
  // collision
      for(i = 0; i <= nxf; i++){ for(j = 0; j <= nyf; j++){
        u2 = uf[i][j]*uf[i][j] + vf[i][j]*vf[i][j];      
        ff0[0][i][j] = rhof[i][j]*(1.0 -3.0/2.0*u2)*4.0/9.0;
        for(k = 1; k <= 4; k++){
          tmp = cx[k]*uf[i][j] + cy[k]*vf[i][j];      
          ff0[k][i][j] = rhof[i][j]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/9.0;
        }  
        for(k = 5; k <= 8; k++){
          tmp = cx[k]*uf[i][j] + cy[k]*vf[i][j];      
          ff0[k][i][j] = rhof[i][j]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/36.0;
        }
      } }

      for(i = 0; i <= nxf; i++){ for(j = 0; j <= nyf; j++){ for(k = 0; k <= 8; k++){
        ff[k][i][j] = ff[k][i][j] - (ff[k][i][j] - ff0[k][i][j])/tauf;
      } } }
 
//..  For Fine --> Coase
      for(i = 0; i <= nxf; i++){ for(k = 0; k <= 8; k++){
         ftc[k][i] =  ff[k][i][nyf - 2];
        ftc0[k][i] = ff0[k][i][nyf - 2];
      } }

    // propagation 
    for(i = 0; i <= nxf; i++){ for(j = 0; j <= nyf; j++){ for(k = 0; k <= 8; k++){
      ftmp[k][i][j] = ff[k][i][j];
    } } }

    for(i = 0; i <= nxf; i++){ for(j = 0; j <= nyf; j++){
      in = i + 1; if(i == nxf){ in =  0;}
      ff[1][in][j ] = ftmp[1][i][j];
    } }
    for(i = 0; i <= nxf; i++){ for(j = 0; j <  nyf; j++){
      jn = j + 1;
      ff[2][i][jn] = ftmp[2][i][j];
    } }
    for(i = 0; i <= nxf; i++){ for(j = 0; j <= nyf; j++){
      in = i - 1; if(i ==   0) {in = nxf;}
      ff[3][in][j] = ftmp[3][i][j];
    } }
    for(i = 0; i <= nxf; i++){ for(j = 1; j <= nyf; j++){
      jn = j - 1;
      ff[8][i][jn] = ftmp[4][i][j];
    } }
    for(i = 0; i <= nxf; i++){ for(j = 0; j <  nyf; j++){
      in = i + 1; if(i == nxf){ in =  0;}
      jn = j + 1;
      ff[5][in][jn] = ftmp[5][i][j]; 
    } }
    for(i = 0; i <= nxf; i++){ for(j = 0; j <  nyf; j++){
      in = i - 1; if(i ==   0) {in = nxf;}
      jn = j + 1;
      ff[6][in][jn] = ftmp[6][i][j];
    } }
    for(i = 0; i <= nxf; i++){ for(j = 1; j <=  nyf; j++){
      in = i - 1; if(i ==   0) {in = nxf;}
      jn = j - 1;
      ff[7][in][jn] = ftmp[7][i][j]; 
    } }
    for(i = 0; i <= nxf; i++){ for(j = 1; j <= nyf; j++){
      in = i + 1; if(i == nxf){ in =  0;}
      jn = j - 1;
      ff[8][in][jn] = ftmp[8][i][j]; 
    } }

    // bounce back condition
    // Non-equilibrium bounce back (Zou)
      for(i = 0; i <= nxf; i++){
        rhof[i][0] = (ff[0][i][0] + ff[1][i][0] + ff[3][i][0]
               + 2.0*(ff[4][i][0] + ff[7][i][0] + ff[8][i][0]))/(1.0 - vb);
        ff[2][i][0] = ff[4][i][0] + rhof[i][0]*vb*2.0/3.0;
        ff[5][i][0] = ff[7][i][0] - 0.5*(ff[1][i][0] - ff[3][i][0])
                    + rhof[i][0]*( ub/2.0 + vb/6.0);
        ff[6][i][0] = ff[8][i][0] + 0.5*(ff[1][i][0] - ff[3][i][0])
                    + rhof[i][0]*(-ub/2.0 + vb/6.0);
      } 

//.. Interface (Coase --> Fine)
      if(n == 0) {
        for(i = 0; i <= nxc; i++){ for(k = 0; k <= 8; k++){
          in  = 2*i;
          ff[k][in][nyf] = ctf02[k][i]
          + (ctf2[k][i] - ctf02[k][i])*(tauf - 1.0)/(tauc - 1.0)/(double)m;
        } }
      } else {
//.. nfine = 2
        for(i = 0; i <= nxc; i++){ for(k = 0; k <= 8; k++){
          fch[k][i] = (ctf3[k][i] - 2.0*ctf2[k][i]  + ctf1[k][i])/8.0
                    + (ctf3[k][i] - ctf1[k][i])/4.0 + ctf2[k][i];
          fc0h[k][i] = (ctf03[k][i] - 2.0*ctf02[k][i]  + ctf01[k][i])/8.0
                     + (ctf03[k][i] - ctf01[k][i])/4.0 + ctf02[k][i];

          in = 2*i;
          ff[k][in][nyf] = fc0h[k][i]
          + (fch[k][i] - fc0h[k][i])*(tauf - 1.0)/(tauc - 1.0)/(double)m;
        } }
      }

// spline
     for(kk = 0; kk <= 8; kk++){
       for(i = 0; i <= nxc; i++){
         in  = 2*i;
         a[i] = ff[kk][in][nyf]; b[i] = 0.0; c[i] = 0.0; d[i] = 0.0;
         for (j = 0; j < nxc; j++){
           mm[i][j]  = 0.0;
         }
       }

       mm[0    ][0    ] = 4.0;  mm[0    ][1    ] = 1.0;
       mm[nxc-1][nxc-2] = 1.0;  mm[nxc-1][nxc-1] = 4.0;

      for (i = 1; i < nxc - 1; i++){
        mm[i-1][i] = 1.0; mm[i][i] = 4.0; mm[i+1][i] = 1.0;
      }

      c[0] = 3.0*(a[1] - 2.0*a[0]);
      for (i = 1; i <= nxc - 1; i++){
        c[i] = 3.0*(a[i+1] - a[i]) - 3.0*(a[i] - a[i-1]);
      }
      for (k = 0; k < nxc - 1; k++){
        for (i = k + 1; i <= nxc - 1; i++){
          r = mm[i][k] / mm[k][k];
          for (j = k; j <= nxc - 1; j++){
            mm[i][j] -= mm[k][j] * r;
          }
          c[i] -= c[k] * r;
        }
      }

      for (i = nxc - 2; i >= 0; i--){
        for (j = i + 1; j < nxc - 1; j++){
          c[i] -= mm[i][j] * c[j];
        }
        c[i] /= mm[i][i];
      }

      for (i = 0; i < nxc; i++){
        b[i] = (a[i+1] - a[i]) - (2.0*c[i+1] + c[i])/3.0;
        d[i] = (c[i+1] - c[i])/3.0;
      }

      for(i = 0; i < nxc; i++){
        in  = 2*i + 1;
        ff[kk][in][nyf] = a[i] + b[i]*0.5 +  c[i] *0.25 + d[i]*0.125;
      }
    }

    // physics
      for(i = 0; i <= nxf; i++){ for(j = 0; j <= nyf; j++){
        rhof[i][j] = ff[0][i][j]; uf[i][j] = 0; vf[i][j] = 0;
        for( k = 1; k <= 8; k++){
          rhof[i][j] = rhof[i][j] + ff[k][i][j];
            uf[i][j] =   uf[i][j] + ff[k][i][j]*cx[k];
            vf[i][j] =   vf[i][j] + ff[k][i][j]*cy[k];
        } 
        uf[i][j] = uf[i][j]/rhof[i][j];
        vf[i][j] = vf[i][j]/rhof[i][j];
      } }
//.. nfine
    }

    norm = 0.0;

    for(i = 0; i <= nxc; i++){ for(j = 0; j <= nyc; j++){
      tmp = sqrt(pow(uc[i][j]-ucn[i][j],2) + pow(vc[i][j]-vcn[i][j],2));
      if(tmp > norm){ norm = tmp; }
    } }
    for(i = 0; i <= nxf; i++){ for(j = 0; j <= nyf; j++){
      tmp = sqrt(pow(uf[i][j]-ufn[i][j],2) + pow(vf[i][j]-vfn[i][j],2));
      if(tmp > norm){ norm = tmp; }
    } }


  } //loop2

//.. coarse + fine --> whole region
  for(i = 0; i <= nx; i++){
    for(j = 0; j <= ny/2; j++){
      rho[i][j] = rhof[i*2][j*2];
        u[i][j] =   uf[i*2][j*2];
        v[i][j] =   vf[i*2][j*2];
    }
    for(j = ny/2 + 1; j <= ny; j++){
      rho[i][j] = rhoc[i   ][j - ny/2 + 1];
        u[i][j] =   uc[i   ][j - ny/2 + 1];
        v[i][j] =   vc[i   ][j - ny/2 + 1];
    } 
  }

  if(time == 100){
    fp = fopen("data100","w");
      for(j = 0; j <= ny; j++){
        fprintf(fp,"%10.8e\n", u[nx/2][j]);
      }
    fclose(fp);

    fp = fopen("data100f","w");
      for(j = 0; j <= nyf; j++){
        fprintf(fp,"%10.8e\n", uf[nxf/2][j]);
      }
    fclose(fp);
  }

  if(time == 200){
    fp = fopen("data200","w");
      for(j = 0; j <= ny; j++){
        fprintf(fp,"%10.8e\n", u[nx/2][j]);
      }
    fclose(fp);

    fp = fopen("data200f","w");
      for(j = 0; j <= nyf; j++){
        fprintf(fp,"%10.8e\n", uf[nxf/2][j]);
      }
    fclose(fp);
  }

  if(time == 500){
    fp = fopen("data500","w");
      for(j = 0; j <= ny; j++){
        fprintf(fp,"%10.8e\n", u[nx/2][j]);
      }
    fclose(fp);

    fp = fopen("data500f","w");
      for(j = 0; j <= nyf; j++){
        fprintf(fp,"%10.8e\n", uf[nxf/2][j]);
      }
    fclose(fp);
  }

  if(time == 1000){
    fp = fopen("data1000","w");
      for(j = 0; j <= ny; j++){
        fprintf(fp,"%10.8e\n", u[nx/2][j]);
      }
    fclose(fp);

    fp = fopen("data1000f","w");
      for(j = 0; j <= nyf; j++){
        fprintf(fp,"%10.8e\n", uf[nxf/2][j]);
      }
    fclose(fp);
  }

  if(time == 3000){
    fp = fopen("data3000","w");
      for(j = 0; j <= ny; j++){
        fprintf(fp,"%10.8e\n", u[nx/2][j]);
      }
    fclose(fp);

    fp = fopen("data3000f","w");
      for(j = 0; j <= nyf; j++){
        fprintf(fp,"%10.8e\n", uf[nxf/2][j]);
      }
    fclose(fp);
  }

  printf("Time = %d, Norm = %15.8e\n", time, norm);
  printf("U[ny    ] = %8.6e(%6.4e)\n",u[nx/2][ny    ], ut);
  printf("U[ny/2+1] = %8.6e(%6.4e)\n",u[nx/2][ny/2+1], ut*(double)(ny/2+1)/(double)ny);
  printf("U[ny/2  ] = %8.6e(%6.4e)\n",u[nx/2][ny/2  ], (ut + ub)*0.5);
  printf("U[ny/2-1] = %8.6e(%6.4e)\n",u[nx/2][ny/2-1], ut*(double)(ny/2-1)/(double)ny);
  printf("U[0     ] = %8.6e(%6.4e)\n",u[nx/2][0     ], ub);

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    aa[i][j]='0';
  } }

  for(j = 0; j <= ny; j++){
    nb = u[nx/2][j]/ut*nx;
    for(i = 0; i <= nb; i++){
      aa[i][j]= '-';
    }
  }

  for(j = ny; j >= 0; j--){ for(i = 0; i <= nx; i++){
    printf("%c",aa[i][j]);
    }
    printf("\n");
  }

  if(norm < 0.0000000001 && time > 10000){
    fp = fopen("data","w");
      for(j = 0; j <= ny; j++){
        fprintf(fp,"%10.8e\n", u[nx/2][j]);
      }
    fclose(fp);

    fp = fopen("dataf","w");
      for(j = 0; j <= nyf; j++){
        fprintf(fp,"%10.8e\n", uf[nxf/2][j]);
      }
    fclose(fp);

    exit(0);
  }

  } //loop1

  return 0;
}
