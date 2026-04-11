// Poiseuille flow 
// Finite Difference Lattice Boltzmann Method.
// The source code is written in c programming language.
// Copyright @2021 Takeshi Seta All Rights Reserved.
//
// 6  2  5
//    |
// 3--0--1
//    |
// 7  4  8
//
// rho: density
// u  : horizontal velocity component (time = n    )
// un : horizontal velocity component (time = n - 1)
// v  : vertical velocity component   (time = n    )
// vn : vertical velocity component   (time = n - 1)
// gx : body force (x-axis)
// gy : body force (y-axis)
// h  : channel width
// dx : grid spacing
// dt : time step
// nx : number of grid points (x-axis)
// ny : number of grid points (y-axis)
// f  : distribution function
// f0 : equilibrium distribution function
// cx : discrete velocity (x-axis)
// cy : discrete velocity (y-axis)
// tau: relaxation time
// nu : kinematic viscosity

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

# define DIM 30

int main(void)
{
  FILE   *fp;
  int    nx = 20, ny = 22, loop1, loop2;
  int    i, j, k, ip, im, nb;
  double u[DIM][DIM], v[DIM][DIM], un[DIM][DIM], vn[DIM][DIM], rho[DIM][DIM];
  double f[9][DIM][DIM], f0[9][DIM][DIM], ftmp[9][DIM][DIM], cx[9], cy[9];
  double gx = 0.00001, gy = 0.0, tmp, u2, nu, norm, time = 0.0;
  double tau = 0.06, dx = 1.0, dy = 1.0, dt = 0.1;
  char   a[DIM][DIM];

  // initial condition
  nu = tau/3.0;

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    u[i][j] = 0.0; v[i][j] = 0.0; un[i][j] = 0.0; vn[i][j] = 0.0; rho[i][j] = 1.0;
  } }

  cx[0] = 0.0; cy[0] = 0.0;
  cx[1] = 1.0; cy[1] = 0.0;  cx[2] = 0.0; cy[2] = 1.0;
  cx[3] =-1.0; cy[3] = 0.0;  cx[4] = 0.0; cy[4] =-1.0;
  cx[5] = 1.0; cy[5] = 1.0;  cx[6] =-1.0; cy[6] = 1.0;
  cx[7] =-1.0; cy[7] =-1.0;  cx[8] = 1.0; cy[8] =-1.0;

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    u2 = u[i][j]*u[i][j] + v[i][j]*v[i][j];      
    f0[0][i][j] = rho[i][j]*(1.0 -3.0/2.0*u2)*4.0/9.0;
    for(k = 1; k <= 4; k++){
      tmp = cx[k]*u[i][j] + cy[k]*v[i][j];      
      f0[k][i][j] = rho[i][j]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/9.0;
    }
    for(k = 5; k <= 8; k++){
      tmp = cx[k]*u[i][j] + cy[k]*v[i][j];      
      f0[k][i][j] = rho[i][j]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/36.0;
    }
  } }

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
    f[k][i][j] = f0[k][i][j];
  } } }

  for(loop1 = 0; loop1 < 200; loop1++){
  for(loop2 = 0; loop2 < 1000; loop2++){
    time = time + dt;

    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      un[i][j] = u[i][j]; vn[i][j] = v[i][j];
    } }

    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
      ftmp[k][i][j] = 0.0;
    } } }

  // collision (SRT)
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      u2 = u[i][j]*u[i][j] + v[i][j]*v[i][j];      
      f0[0][i][j] = rho[i][j]*(1.0 -3.0/2.0*u2)*4.0/9.0;
      for(k = 1; k <= 4; k++){
        tmp = cx[k]*u[i][j] + cy[k]*v[i][j];      
        f0[k][i][j] = rho[i][j]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/9.0;
      }  
      for(k = 5; k <= 8; k++){
        tmp = cx[k]*u[i][j] + cy[k]*v[i][j];      
        f0[k][i][j] = rho[i][j]*(1.0 +3.0*tmp +9.0/2.0*tmp*tmp -3.0/2.0*u2)/36.0;
      }
    } }

    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
      ftmp[k][i][j] = ftmp[k][i][j] - (f[k][i][j] - f0[k][i][j])/tau;
    } } }

    // force
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      for(k = 1; k <= 4; k++){
        ftmp[k][i][j] = ftmp[k][i][j] + rho[i][j]*(cx[k]*gx + cy[k]*gy)/3.0;
      }
      for(k = 5; k <= 8; k++){
        ftmp[k][i][j] = ftmp[k][i][j] + rho[i][j]*(cx[k]*gx + cy[k]*gy)/12.0;
      }
    } }

    // advection term (FTCS)
    for(i = 0; i <= nx; i++){ for(j = 1; j <= ny-1; j++){ for(k = 0; k <= 8; k++){
      ip = i + 1; if(i == nx){ip =  0;}
      im = i - 1; if(i ==  0){im = nx;}
      ftmp[k][i][j] = ftmp[k][i][j] - cx[k]*(f[k][ip][j  ] - f[k][im][j  ])/dx*0.5 
                                    - cy[k]*(f[k][i ][j+1] - f[k][i ][j-1])/dy*0.5; 
    } } }

    // boundary condition (mesoscale, extrapolation)
    for(i = 0; i <= nx; i++){ for(k = 0; k <= 8; k++){
      ftmp[k][i][ 0] = 2.0*ftmp[k][i][   1] - ftmp[k][i][   2];
      ftmp[k][i][ny] = 2.0*ftmp[k][i][ny-1] - ftmp[k][i][ny-2];
    } }

    // propagation (FTCS)
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){ for(k = 0; k <= 8; k++){
      f[k][i][j] = f[k][i][j] + ftmp[k][i][j]*dt;
    } } }

    // physics
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      rho[i][j] = f[0][i][j]; u[i][j] = 0; v[i][j] = 0;
      for( k = 1; k <= 8; k++){
        rho[i][j] = rho[i][j] + f[k][i][j];
          u[i][j] =   u[i][j] + f[k][i][j]*cx[k];
          v[i][j] =   v[i][j] + f[k][i][j]*cy[k];
      } 
      u[i][j] = u[i][j]/rho[i][j];
      v[i][j] = v[i][j]/rho[i][j];
    } }

    // boundary condition (macroscale)
    for(i = 0; i <= nx; i++){
      u[i][   1] = 0.0; v[i][   1] = 0.0;
      u[i][ny-1] = 0.0; v[i][ny-1] = 0.0;
    }

    norm = 0.0;
    for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
      tmp = sqrt(pow(u[i][j]-un[i][j],2) + pow(v[i][j]-vn[i][j],2));
      if(tmp > norm){ norm = tmp; }
    } }

  } //loop2

  printf("Time = %10.2f, Norm = %8.6e\n", time, norm);
  printf("Umax = %8.6e(%6.4e)\n",u[nx/2][ny/2], gx/8.0/nu*(ny-2)*(ny-2));

  for(i = 0; i <= nx; i++){ for(j = 0; j <= ny; j++){
    a[i][j]='0';
  } }

  for(j = 0; j <= ny; j++){
    nb = u[nx/2][j]/(gx/8.0/nu*(ny-2)*(ny-2))*20;
    for(i = 0; i <= nb; i++){
      a[i][j]= '-';
    }
  }

  for(j = 0; j <= ny; j++){ for(i = 0; i <= nx; i++){
    printf("%c",a[i][j]);
    }
    printf("\n");
  }

  if(norm < 0.0000000001 && time > 10000){ 
    fp = fopen("data","w");

    for(j = 0; j <= ny; j++){
      fprintf(fp,"%10.8e\n", u[nx/2][j]/(gx/8.0/nu*(ny-2)*(ny-2)));
    }
    fclose(fp);

    exit(0);
  }

  } //loop1

  return 0;
}
