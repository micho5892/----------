// Natural Convection
// double-population thermal lattice Boltzmann method.
// The source code is written in C programming language.
// Copyright @2021 Takeshi Seta All Rights Reserved.
//
// (f) 6  2  5     (g)   2
//        |              | 
//     3--0--1        3--0--1
//        |              |
//     7  4  8           4
//
// rho   : density
// u     : horizontal velocity component (time = n)
// v     : vertical velocity component (time = n)
// e     : temperature (numerical solution, time = n    )
// un    : horizontal velocity component (time = n -1)
// vn    : vertical velocity component (time = n - 1)
//
// nx    : number of grid points (x-axis)
// ny    : number of grid points (y-axis)
// h     : channel width
// fx    : body force (x-axis)
// fy    : body force (y-axis)
// rbetag: coefficient of thermal expansion multiplied by gravity
//         (Boussinesq approximation) 
// nu    : kinematic viscosity
// chi   : diffusion coefficient
//
// f     : distribution function (density, velocity)
// f0    : equilibrium distribution function (density, velocity)
// g     : distribution function (temperature)
// g0    : equilibrium distribution function (temperature)
// cx    : discrete velocity (x-axis)
// cy    : discrete velocity (y-axis)
// tauf  : relaxation time (f)
// taug  : relaxation time (g)
//
// mmu, mme: transformation matrices
// ui , ei : inverse matrices of the transformation matrices
// sf , sg : relaxation matrices
// msf     : ui*sf
// msg     : ei*sg
// re      : Reynolds number
// ra      : Rayleigh number
// pr      : Prandtl number
// flag = 0 (SRT collision operator)
// flag = 1 (MRT collision operator)

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

# define DIM 48 

int main(void)
{
  FILE   *fp;
  int    nx = 46, ny = 46, time = 0, loop1, loop2;
  int    i, j, k, m, ip, im, jp, jm, flag;
  double rho[DIM][DIM], u[DIM][DIM], v[DIM][DIM], e[DIM][DIM];
  double  un[DIM][DIM], vn[DIM][DIM];
  double  fx[DIM][DIM], fy[DIM][DIM];
  double f[9][DIM][DIM], f0[9][DIM][DIM], ftmp[9][DIM][DIM], cx[9], cy[9];
  double g[5][DIM][DIM], g0[5][DIM][DIM], gtmp[5][DIM][DIM];

  double mf[9][DIM][DIM], mf0[9][DIM][DIM], mcf[9][DIM][DIM];
  double mg[5][DIM][DIM], mg0[5][DIM][DIM], mcg[5][DIM][DIM];
  double mmu[9][9], ui[9][9], sf[9][9], msf[9][9];
  double mme[5][5], ei[5][5], sg[5][5], msg[5][5];
  double umax, umin, tmp, u2, nu, chi, norm, tauf, taug, rbetag, h, ra, pr;
  char   a[DIM][DIM];

  // initial condition
  pr = 0.71;
  ra =   10000.0;
  h = (double)(nx - 1);
  tauf = 0.8;
  nu = (tauf - 0.5)/3.0;
  chi = nu/pr;
  taug = 3.0*chi + 0.5;
  rbetag = ra*nu*chi/h/h/h;
  flag = 0; // SRT
  flag = 1; // MRT 

//.. collision matrix (f)
  mmu[0][0]= 1.0; mmu[0][1]= 1.0; mmu[0][2]= 1.0;
  mmu[0][3]= 1.0; mmu[0][4]= 1.0; mmu[0][5]= 1.0;
  mmu[0][6]= 1.0; mmu[0][7]= 1.0; mmu[0][8]= 1.0;

  mmu[1][0]=-4.0; mmu[1][1]=-1.0; mmu[1][2]=-1.0;
  mmu[1][3]=-1.0; mmu[1][4]=-1.0; mmu[1][5]= 2.0;
  mmu[1][6]= 2.0; mmu[1][7]= 2.0; mmu[1][8]= 2.0;

  mmu[2][0]= 4.0; mmu[2][1]=-2.0; mmu[2][2]=-2.0;
  mmu[2][3]=-2.0; mmu[2][4]=-2.0; mmu[2][5]= 1.0;
  mmu[2][6]= 1.0; mmu[2][7]= 1.0; mmu[2][8]= 1.0;

  mmu[3][0]= 0.0; mmu[3][1]= 1.0; mmu[3][2]= 0.0;
  mmu[3][3]=-1.0; mmu[3][4]= 0.0; mmu[3][5]= 1.0;
  mmu[3][6]=-1.0; mmu[3][7]=-1.0; mmu[3][8]= 1.0;

  mmu[4][0]= 0.0; mmu[4][1]=-2.0; mmu[4][2]= 0.0;
  mmu[4][3]= 2.0; mmu[4][4]= 0.0; mmu[4][5]= 1.0;
  mmu[4][6]=-1.0; mmu[4][7]=-1.0; mmu[4][8]= 1.0;

  mmu[5][0]= 0.0; mmu[5][1]= 0.0; mmu[5][2]= 1.0;
  mmu[5][3]= 0.0; mmu[5][4]=-1.0; mmu[5][5]= 1.0;
  mmu[5][6]= 1.0; mmu[5][7]=-1.0; mmu[5][8]=-1.0;

  mmu[6][0]= 0.0; mmu[6][1]= 0.0; mmu[6][2]=-2.0;
  mmu[6][3]= 0.0; mmu[6][4]= 2.0; mmu[6][5]= 1.0;
  mmu[6][6]= 1.0; mmu[6][7]=-1.0; mmu[6][8]=-1.0;

  mmu[7][0]= 0.0; mmu[7][1]= 1.0; mmu[7][2]=-1.0;
  mmu[7][3]= 1.0; mmu[7][4]=-1.0; mmu[7][5]= 0.0;
  mmu[7][6]= 0.0; mmu[7][7]= 0.0; mmu[7][8]= 0.0;
  
  mmu[8][0]= 0.0; mmu[8][1]= 0.0; mmu[8][2]= 0.0;
  mmu[8][3]= 0.0; mmu[8][4]= 0.0; mmu[8][5]= 1.0;
  mmu[8][6]=-1.0; mmu[8][7]= 1.0; mmu[8][8]=-1.0;

//
  ui[0][0]= 1.0/9.0 ; ui[0][1]= -1.0/9.0; ui[0][2]= 1.0/9.0 ;
  ui[0][3]= 0.0     ; ui[0][4]= 0.0     ; ui[0][5]= 0.0     ;
  ui[0][6]= 0.0     ; ui[0][7]= 0.0     ; ui[0][8]= 0.0     ;

  ui[1][0]= 1.0/9.0 ; ui[1][1]=-1.0/36.0; ui[1][2]=-1.0/18.0;
  ui[1][3]= 1.0/6.0 ; ui[1][4]=-1.0/6.0 ; ui[1][5]= 0.0     ;
  ui[1][6]= 0.0     ; ui[1][7]= 1.0/4.0 ; ui[1][8]= 0.0     ;

  ui[2][0]= 1.0/9.0 ; ui[2][1]=-1.0/36.0; ui[2][2]=-1.0/18.0;
  ui[2][3]= 0.0     ; ui[2][4]= 0.0     ; ui[2][5]= 1.0/6.0 ;
  ui[2][6]=-1.0/6.0 ; ui[2][7]=-1.0/4.0 ; ui[2][8]= 0.0     ;

  ui[3][0]= 1.0/9.0 ; ui[3][1]=-1.0/36.0; ui[3][2]=-1.0/18.0;
  ui[3][3]=-1.0/6.0 ; ui[3][4]= 1.0/6.0 ; ui[3][5]= 0.0     ;
  ui[3][6]= 0.0     ; ui[3][7]= 1.0/4.0 ; ui[3][8]= 0.0     ;

  ui[4][0]= 1.0/9.0 ; ui[4][1]=-1.0/36.0; ui[4][2]=-1.0/18.0;
  ui[4][3]= 0.0     ; ui[4][4]= 0.0     ; ui[4][5]=-1.0/6.0 ;
  ui[4][6]= 1.0/6.0 ; ui[4][7]=-1.0/4.0 ; ui[4][8]= 0.0     ;

  ui[5][0]= 1.0/9.0 ; ui[5][1]= 1.0/18.0; ui[5][2]= 1.0/36.0;
  ui[5][3]= 1.0/6.0 ; ui[5][4]= 1.0/12.0; ui[5][5]= 1.0/6.0 ;
  ui[5][6]= 1.0/12.0; ui[5][7]= 0.0     ; ui[5][8]= 1.0/4.0 ;

  ui[6][0]= 1.0/9.0 ; ui[6][1]= 1.0/18.0; ui[6][2]= 1.0/36.0;
  ui[6][3]=-1.0/6.0 ; ui[6][4]=-1.0/12.0; ui[6][5]= 1.0/6.0 ;
  ui[6][6]= 1.0/12.0; ui[6][7]= 0.0     ; ui[6][8]=-1.0/4.0 ;

  ui[7][0]= 1.0/9.0 ; ui[7][1]= 1.0/18.0; ui[7][2]= 1.0/36.0;
  ui[7][3]=-1.0/6.0 ; ui[7][4]=-1.0/12.0; ui[7][5]=-1.0/6.0 ;
  ui[7][6]=-1.0/12.0; ui[7][7]= 0.0     ; ui[7][8]= 1.0/4.0 ;
  
  ui[8][0]= 1.0/9.0 ; ui[8][1]= 1.0/18.0; ui[8][2]= 1.0/36.0;
  ui[8][3]= 1.0/6.0 ; ui[8][4]= 1.0/12.0; ui[8][5]=-1.0/6.0 ;
  ui[8][6]=-1.0/12.0; ui[8][7]= 0.0     ; ui[8][8]=-1.0/4.0 ;

  for(i = 0; i <= 8; i++){ for(j = 0; j <= 8; j++){
    sf[i][j] = 0.0;
  } }

  sf[0][0] = 0.0; sf[1][1] = 1.5     ; sf[2][2] = 1.4     ;
  sf[3][3] = 0.0; sf[4][4] = 1.5     ; sf[5][5] = 0.0     ;
  sf[6][6] = 1.5; sf[7][7] = 1.0/tauf; sf[8][8] = 1.0/tauf;

  for(i = 0; i <= 8; i++){ for(j = 0; j <= 8; j++){
    msf[i][j] = 0.0;
    for(k = 0; k <= 8; k++){
      msf[i][j] = msf[i][j] + ui[i][k]*sf[k][j];
    }
  } }

//.. collision matrix (g)
  mme[0][0]= 1.0;mme[0][1]= 1.0;mme[0][2]= 1.0;mme[0][3]= 1.0;mme[0][4]= 1.0;
  mme[1][0]= 0.0;mme[1][1]= 1.0;mme[1][2]= 0.0;mme[1][3]=-1.0;mme[1][4]= 0.0;
  mme[2][0]= 0.0;mme[2][1]= 0.0;mme[2][2]= 1.0;mme[2][3]= 0.0;mme[2][4]=-1.0;
  mme[3][0]= 4.0;mme[3][1]=-1.0;mme[3][2]=-1.0;mme[3][3]=-1.0;mme[3][4]=-1.0;
  mme[4][0]= 0.0;mme[4][1]= 1.0;mme[4][2]=-1.0;mme[4][3]= 1.0;mme[4][4]=-1.0;

  ei[0][0]=0.2;ei[0][1]= 0.0;ei[0][2]= 0.0;ei[0][3]= 0.20;ei[0][4]= 0.00;
  ei[1][0]=0.2;ei[1][1]= 0.5;ei[1][2]= 0.0;ei[1][3]=-0.05;ei[1][4]= 0.25;
  ei[2][0]=0.2;ei[2][1]= 0.0;ei[2][2]= 0.5;ei[2][3]=-0.05;ei[2][4]=-0.25;
  ei[3][0]=0.2;ei[3][1]=-0.5;ei[3][2]= 0.0;ei[3][3]=-0.05;ei[3][4]= 0.25;
  ei[4][0]=0.2;ei[4][1]= 0.0;ei[4][2]=-0.5;ei[4][3]=-0.05;ei[4][4]=-0.25;

  for(i = 0; i <= 4; i++){ for(j = 0; j <= 4; j++){
    sg[i][j]  = 0.0;
  } }
  sg[0][0] = 0.0; sg[1][1] = 1.0/taug; sg[2][2] = 1.0/taug; sg[3][3] = 1.0; sg[4][4] = 1.0;

  for(i = 0; i <= 4; i++){ for(j = 0; j <= 4; j++){
     msg[i][j] = 0.0;
     for(k = 0; k <= 4; k++){
       msg[i][j] = msg[i][j] + ei[i][k] * sg[k][j];
     }
   } }

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    u[i][j] = 0.0; v[i][j] = 0.0; un[i][j] = 0.0; vn[i][j] = 0.0;
    rho[i][j] = 1.0;
    e[i][j] = (double)(nx - i)/(double)(nx - 1);
  } }

  cx[0] = 0.0; cy[0] = 0.0;
  cx[1] = 1.0; cy[1] = 0.0;  cx[2] = 0.0; cy[2] = 1.0;
  cx[3] =-1.0; cy[3] = 0.0;  cx[4] = 0.0; cy[4] =-1.0;
  cx[5] = 1.0; cy[5] = 1.0;  cx[6] =-1.0; cy[6] = 1.0;
  cx[7] =-1.0; cy[7] =-1.0;  cx[8] = 1.0; cy[8] =-1.0;

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    u2 = u[i][j]*u[i][j] + v[i][j]*v[i][j];      
    f0[0][i][j] = rho[i][j]*(1.0 -3.0/2.0*u2)*4.0/9.0;
    g0[0][i][j] =   e[i][j]/3.0;
    for(k = 1; k <= 4; k++){
      tmp = cx[k]*u[i][j] + cy[k]*v[i][j];      
      f0[k][i][j] = rho[i][j]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/9.0;
      g0[k][i][j] =   e[i][j]*(1.0 + 3.0*tmp)/6.0;
    }
    for(k = 5; k <= 8; k++){
      tmp = cx[k]*u[i][j] + cy[k]*v[i][j];      
      f0[k][i][j] = rho[i][j]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/36.0;
    }
  } }

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    for(k = 0; k <= 8; k++){
      f[k][i][j] = f0[k][i][j];
    }
    for(k = 0; k <= 4; k++){
      g[k][i][j] = g0[k][i][j];
    }
  } }

// calculation start
  for(loop1 = 0; loop1 < 50; loop1++){
  for(loop2 = 0; loop2 < 100; loop2++){
    time++;

    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      un[i][j] = u[i][j]; vn[i][j] = v[i][j];
    } }

  // collision
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      u2 = u[i][j]*u[i][j] + v[i][j]*v[i][j];      
      f0[0][i][j] = rho[i][j]*(1.0 -3.0/2.0*u2)*4.0/9.0;
      g0[0][i][j] =   e[i][j]/3.0;
      for(k = 1; k <= 4; k++){
        tmp = cx[k]*u[i][j] + cy[k]*v[i][j];      
        f0[k][i][j] = rho[i][j]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/9.0;
        g0[k][i][j] =   e[i][j]*(1.0 + 3.0*tmp)/6.0;
      }  
      for(k = 5; k <= 8; k++){
        tmp = cx[k]*u[i][j] + cy[k]*v[i][j];      
        f0[k][i][j] = rho[i][j]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/36.0;
      }
    } }

    if(flag == 0){ 
// SRT collision
      for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
        for(k = 0; k <= 8; k++){
          f[k][i][j] = f[k][i][j] - (f[k][i][j] - f0[k][i][j])/tauf;
        }
        for(k = 0; k <= 4; k++){
          g[k][i][j] = g[k][i][j] - (g[k][i][j] - g0[k][i][j])/taug;
        }
      } }
    }else
    {
//.. MRT collision
      for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
        for(k = 0; k <= 8; k++){
          mf[k][i][j] = 0.0; mf0[k][i][j] = 0.0;
          for(m = 0; m <= 8; m++){
             mf[k][i][j] =  mf[k][i][j] + mmu[k][m]* f[m][i][j];
            mf0[k][i][j] = mf0[k][i][j] + mmu[k][m]*f0[m][i][j];
          }
        }
        for(k = 0; k <= 4; k++){
          mg[k][i][j] = 0.0; mg0[k][i][j]= 0.0;
          for(m = 0; m <= 4; m++){
            mg[k][i][j] =  mg[k][i][j] + mme[k][m]* g[m][i][j];
           mg0[k][i][j] = mg0[k][i][j] + mme[k][m]*g0[m][i][j];
          }
        }
      } }

      for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
        for(k = 0; k <= 8; k++){
          mcf[k][i][j] = 0.0;
          for(m = 0; m <= 8; m++){
            mcf[k][i][j] = mcf[k][i][j] + msf[k][m]*(mf[m][i][j] - mf0[m][i][j]);
          }
        }
        for(k = 0; k <= 4; k++){
          mcg[k][i][j] = 0.0;
          for(m = 0; m <= 4; m++){
            mcg[k][i][j] = mcg[k][i][j] + msg[k][m]*(mg[m][i][j] - mg0[m][i][j]);
          }
        }
      } }

      for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
        for(k = 0; k <= 8; k++){
          f[k][i][j] = f[k][i][j] - mcf[k][i][j];
        }
        for(k = 0; k <= 4; k++){
          g[k][i][j] = g[k][i][j] - mcg[k][i][j];
        }
      } }
    }
  // gravity
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      fx[i][j] = 0.0; fy[i][j] = rbetag*(e[i][j] - 0.5);
    } }

  // forcing
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      for(k = 0; k <= 4; k++){
        f[k][i][j] = f[k][i][j] + (cx[k]*fx[i][j] + cy[k]*fy[i][j])/3.0;
      }
      for(k = 5; k <= 8; k++){
        f[k][i][j] = f[k][i][j] + (cx[k]*fx[i][j] + cy[k]*fy[i][j])/12.0;
      }
    } }

    // propagation 
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      for(k = 0; k <= 8; k++){ ftmp[k][i][j] = f[k][i][j]; }
      for(k = 0; k <= 4; k++){ gtmp[k][i][j] = g[k][i][j]; }
    } }

    for(j = 0; j <= ny; j++){
      for(i = 0; i <= nx-1; i++){ f[1][i+1][j  ] = ftmp[1][i][j];
                                  g[1][i+1][j  ] = gtmp[1][i][j];}
      for(i = 1; i <= nx  ; i++){ f[3][i-1][j  ] = ftmp[3][i][j];
                                  g[3][i-1][j  ] = gtmp[3][i][j];}
    }

    for(j = 0; j <= ny-1; j++){
      for(i = 0; i <= nx  ; i++){ f[2][i  ][j+1] = ftmp[2][i][j];
                                  g[2][i  ][j+1] = gtmp[2][i][j]; }
      for(i = 0; i <= nx-1; i++){ f[5][i+1][j+1] = ftmp[5][i][j]; }
      for(i = 1; i <= nx  ; i++){ f[6][i-1][j+1] = ftmp[6][i][j]; }
    }
    for(j = 1; j <= ny; j++){
      for(i = 0; i <= nx  ; i++){ f[4][i  ][j-1] = ftmp[4][i][j];
                                  g[4][i  ][j-1] = gtmp[4][i][j]; }
      for(i = 1; i <= nx  ; i++){ f[7][i-1][j-1] = ftmp[7][i][j]; }
      for(i = 0; i <= nx-1; i++){ f[8][i+1][j-1] = ftmp[8][i][j]; }
    }

    // boundary condition
    // Half-way bounce back
    for(j = 1; j < ny; j++){
      jm = j - 1; jp = j + 1;
      f[1][   1][j] = f[3][0][j ];
      f[5][   1][j] = f[7][0][jm];
      f[8][   1][j] = f[6][0][jp];

      f[3][nx-1][j] = f[1][nx][j ];
      f[7][nx-1][j] = f[5][nx][jp];
      f[6][nx-1][j] = f[8][nx][jm];

      g[1][   1][j] =-g[3][ 0][j ] + 1.0/3.0;
      g[3][nx-1][j] =-g[1][nx][j ];

    }

    for(i = 1; i < nx; i++){
      im = i - 1; ip = i + 1;
      f[2][i][   1] = f[4][i ][ 0];
      f[5][i][   1] = f[7][im][ 0];
      f[6][i][   1] = f[8][ip][ 0];

      f[4][i][ny-1] = f[2][i ][ny];
      f[7][i][ny-1] = f[5][ip][ny];
      f[8][i][ny-1] = f[6][im][ny];

      g[2][i][   1] = g[4][i ][ 0];
      g[4][i][ny-1] = g[2][i ][ny];
    }
    // physics
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      rho[i][j] = f[0][i][j]; u[i][j] = 0; v[i][j] = 0;
        e[i][j] = g[0][i][j];
      for( k = 1; k <= 4; k++){
        e[i][j] =   e[i][j] + g[k][i][j];
      } 
      for( k = 1; k <= 8; k++){
        rho[i][j] = rho[i][j] + f[k][i][j];
          u[i][j] =   u[i][j] + f[k][i][j]*cx[k];
          v[i][j] =   v[i][j] + f[k][i][j]*cy[k];
      } 
      u[i][j] = u[i][j]/rho[i][j];
      v[i][j] = v[i][j]/rho[i][j];
    } }

    norm = 0.0;
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      tmp = sqrt(pow(u[i][j] - un[i][j],2) + pow(v[i][j] - vn[i][j],2));
      if(tmp > norm){ norm = tmp; }
    } }

  } //loop2

  if(flag == 0){ printf("SRT collision operator\n"); };
  if(flag == 1){ printf("MRT collision operator\n"); };
  printf("Ra = %5.3f, Pr = %5.3f\n", ra, pr);
  printf("Time = %f, Norm = %8.6e\n", (double)time, norm);

  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    a[i][j]='0';
  } }

  umax = -999.9; umin = 999.9;
  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    tmp = e[i][j];
    if(tmp >= umax){umax = tmp;}
    if(tmp <= umin){umin = tmp;}
  } }

  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    tmp = e[i][j];
    if(tmp <=  umax*1.0            ){a[i][j] = '9';}
    if(tmp <= (umax*0.9 + umin*0.1)){a[i][j] = '8';}
    if(tmp <= (umax*0.8 + umin*0.2)){a[i][j] = '7';}
    if(tmp <= (umax*0.7 + umin*0.3)){a[i][j] = '6';}
    if(tmp <= (umax*0.6 + umin*0.4)){a[i][j] = '5';}
    if(tmp <= (umax*0.5 + umin*0.5)){a[i][j] = '4';}
    if(tmp <= (umax*0.4 + umin*0.6)){a[i][j] = '3';}
    if(tmp <= (umax*0.3 + umin*0.7)){a[i][j] = '2';}
    if(tmp <= (umax*0.2 + umin*0.8)){a[i][j] = '1';}
    if(tmp <= (umax*0.1 + umin*0.9)){a[i][j] = '0';}
  } }

  umax = -999.9; umin = 999.9;
  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    tmp = fabs(u[i][j])/chi*h;
    if(tmp >= umax){umax = tmp;}
  } }
  printf("maximum value of horizontal velocity: %10.8f\n", umax);

  umax = -999.9; umin = 999.9;
  for(i = 1; i <= nx - 1; i++){ for(j = 1; j <= ny - 1; j++){
    tmp = fabs(v[i][j])/chi*h;
    if(tmp >= umax){umax = tmp;}
  } }
  printf("maximum value of vertical velocity  : %10.8f", umax);

  for(j = ny; j >= 1; j = j - 2){
    for(i = 1; i <= nx - 1; i = i + 2){
      printf("%c",a[i][j]);
    }
     printf("\n");
  }

  } //loop1

  fp = fopen("datancu","w");
    for(j = 1; j <= ny; j=j+2){ for(i = 1; i <= nx; i=i+2){
      fprintf(fp," %10.8e", u[i][j]/chi*h);
    }
  fprintf(fp,"\n");
  } 
  fclose(fp);

  fp = fopen("datancv","w");
    for(j = 1; j <= ny; j=j+2){ for(i = 1; i <= nx; i=i+2){
      fprintf(fp," %10.8e", v[i][j]/chi*h);
    }
  fprintf(fp,"\n");
  } 
  fclose(fp);

  fp = fopen("datance","w");
    for(j = 1; j <= ny; j=j+2){ for(i = 1; i <= nx; i=i+2){
      fprintf(fp," %10.8e", e[i][j]);
    }
  fprintf(fp,"\n");
  } 
  fclose(fp);

  return 0;
}
